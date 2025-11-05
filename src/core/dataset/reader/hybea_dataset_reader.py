"""Reader for HybEA-formatted datasets."""

from pathlib import Path
import unicodedata
from typing import Dict, Iterable, Optional, Set, Tuple

from rdflib import URIRef

from src.core.dataset import Dataset
from src.core.dataset.reader.dataset_reader_base import DatasetReader
from src.core.knowledge_graph.reader import KnowledgeGraphReaderFactory
from src.logger import get_logger
from src.util.reader import read_tsv

logger = get_logger(__name__)


class HybeaDatasetReader(DatasetReader):
    """Assemble a `Dataset` object from HybEA/KnowFormer data layouts."""

    file_type = "hybea"

    ATTR_ENT_IDS_1 = "ent_ids_1"
    ATTR_ENT_IDS_2 = "ent_ids_2"
    ATTR_REF_PAIRS = "ref_pairs"
    ATTR_SUP_PAIRS = "sup_pairs"
    ATTR_VALID_PAIRS = "valid_pairs"

    KF_ILL = "ent_ILLs.txt"

    def read(self, dir_path: str, **_) -> Dataset:
        """Load knowledge graphs plus aligned entity pairs from a directory tree.

        The subtype (attribute_data/knowformer_data) is inferred from the path structure:
        - If dir_path ends with attribute_data or knowformer_data, use that directly
        - Otherwise, search for subdirectories with those names
        """
        base_path = Path(dir_path)
        dataset_name = (
            base_path.parent.name
            if base_path.name in {"attribute_data", "knowformer_data"}
            else base_path.name
        )

        # Infer subtype from path instead of using parameter
        subtype = None
        if base_path.name in {"attribute_data", "knowformer_data"}:
            subtype = base_path.name

        variant_dirs = self._gather_variant_dirs(base_path, subtype)
        datasets = [(variant.name, self._load_dataset(variant)) for variant in variant_dirs]
        chosen = self._choose_dataset(dataset_name, datasets)
        return chosen

    def _gather_variant_dirs(self, base_path: Path, subtype: Optional[str]) -> Tuple[Path, ...]:
        variants = []

        def add_if_exists(path: Path):
            if path.exists() and path not in variants:
                variants.append(path)

        if base_path.name in {"attribute_data", "knowformer_data"}:
            add_if_exists(base_path)
            sibling = base_path.parent / (
                "knowformer_data" if base_path.name == "attribute_data" else "attribute_data"
            )
            add_if_exists(sibling)
        else:
            if subtype:
                add_if_exists(base_path / subtype)
                if variants:
                    return tuple(variants)
            add_if_exists(base_path / "attribute_data")
            add_if_exists(base_path / "knowformer_data")
            if not variants:
                add_if_exists(base_path)

        return tuple(variants) if variants else (base_path,)

    def _load_dataset(self, data_dir: Path) -> Dataset:
        kg_reader = KnowledgeGraphReaderFactory.create_reader(self.file_type)
        subtype = data_dir.name if data_dir.name in {"attribute_data", "knowformer_data"} else None

        kg1 = kg_reader.read(str(data_dir), kg_number=1, subtype=subtype)
        kg2 = kg_reader.read(str(data_dir), kg_number=2, subtype=subtype)

        if self._is_attribute_dir(data_dir):
            aligned_entities = self._aligned_entities_attribute_data(data_dir)
        else:
            aligned_entities = self._aligned_entities_knowformer_data(data_dir)

        return Dataset(kg1, kg2, aligned_entities)

    def _choose_dataset(self, dataset_name: str, datasets: Iterable[Tuple[str, Dataset]]) -> Dataset:
        datasets = list(datasets)
        if len(datasets) == 1:
            return datasets[0][1]

        summaries = []
        for label, ds in datasets:
            summary = {
                "source_triples": len(ds.knowledge_graph_source),
                "target_triples": len(ds.knowledge_graph_target),
                "aligned_pairs": len(ds.aligned_entities),
            }
            summaries.append((label, ds, summary))

        base_summary = summaries[0][2]
        for label, _, summary in summaries[1:]:
            if summary != base_summary:
                logger.warning(
                    "HybEA dataset '%s': mismatch between variants. baseline=%s, %s=%s",
                    dataset_name,
                    base_summary,
                    label,
                    summary,
                )

        chosen_label, chosen_ds, chosen_summary = max(
            summaries,
            key=lambda item: item[2]["source_triples"] + item[2]["target_triples"],
        )
        logger.info(
            "HybEA dataset '%s': selected variant '%s' with summary %s",
            dataset_name,
            chosen_label,
            chosen_summary,
        )
        return chosen_ds

    def _is_attribute_dir(self, data_dir: Path) -> bool:
        """Decide whether the directory represents attribute-style data."""
        if data_dir.name == "attribute_data":
            return True
        if data_dir.name == "knowformer_data":
            return False
        return (data_dir / self.ATTR_REF_PAIRS).exists()

    def _load_entity_ids(self, ent_ids_1_path: Path, ent_ids_2_path: Path) -> Dict[str, str]:
        """Build a lookup map from normalized entity IDs to their URI representation."""
        ent_ids: Dict[str, str] = {}
        for rows in (read_tsv(ent_ids_1_path), read_tsv(ent_ids_2_path)):
            for src, dst in rows:
                nsrc = unicodedata.normalize("NFC", src)
                ndst = unicodedata.normalize("NFC", dst)
                ent_ids[nsrc] = ndst
        return ent_ids

    def _collect_aligned_pairs(
        self, ent_ids: Dict[str, str], pair_files: Iterable[Path]
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Return aligned entity pairs after normalizing all identifiers."""

        aligned: Set[Tuple[URIRef, URIRef]] = set()
        for file_path in pair_files:
            for left, right in read_tsv(file_path):
                nleft = unicodedata.normalize("NFC", left)
                nright = unicodedata.normalize("NFC", right)
                if nleft in ent_ids and nright in ent_ids:
                    aligned.add((URIRef(ent_ids[nleft]), URIRef(ent_ids[nright])))

        return aligned

    def _aligned_entities_attribute_data(self, data_dir: Path) -> Set[Tuple[URIRef, URIRef]]:
        """Load aligned entity pairs for the attribute-style dataset layout."""
        ent_ids_1_path = data_dir / self.ATTR_ENT_IDS_1
        ent_ids_2_path = data_dir / self.ATTR_ENT_IDS_2
        ent_ids = self._load_entity_ids(ent_ids_1_path, ent_ids_2_path)

        pair_files = (
            data_dir / self.ATTR_REF_PAIRS,
            data_dir / self.ATTR_SUP_PAIRS,
            data_dir / self.ATTR_VALID_PAIRS,
        )
        return self._collect_aligned_pairs(ent_ids, pair_files)

    def _aligned_entities_knowformer_data(self, data_dir: Path) -> Set[Tuple[URIRef, URIRef]]:
        """Load aligned entities for the KnowFormer-format dataset layout."""

        pair_files = (data_dir / self.KF_ILL,)

        aligned: Set[Tuple[URIRef, URIRef]] = set()
        for file_path in pair_files:
            for left, right in read_tsv(file_path):
                nleft = unicodedata.normalize("NFC", left)
                nright = unicodedata.normalize("NFC", right)
                aligned.add((URIRef(nleft), URIRef(nright)))

        return aligned
