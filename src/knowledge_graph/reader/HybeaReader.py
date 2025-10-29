"""Knowledge graph reader for HybEA datasets."""

from pathlib import Path
from typing import Iterable, Optional, Tuple

from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.knowledge_graph.reader.Reader import Reader
from src.logger import get_logger
from src.util.reader import read_tsv

logger = get_logger(__name__)


class HybeaReader(Reader):
    """Load individual knowledge graphs from HybEA attribute or KnowFormer layouts."""

    file_type = "hybea"

    def read(self, dir_path, kg_number=None, subtype: Optional[str] = None, **_) -> KnowledgeGraph:
        base_path = Path(dir_path)
        dataset_name = (
            base_path.parent.name
            if base_path.name in {"attribute_data", "knowformer_data"}
            else base_path.name
        )

        variant_dirs = self._gather_variant_dirs(base_path, subtype)
        graphs = [(variant.name, self._load_graph(variant, kg_number)) for variant in variant_dirs]
        chosen = self._choose_graph(dataset_name, kg_number, graphs)
        return chosen

    def _gather_variant_dirs(self, base_path: Path, subtype: Optional[str]) -> Tuple[Path, ...]:
        variants = []

        def add(path: Path):
            if path.exists() and path not in variants:
                variants.append(path)

        if base_path.name in {"attribute_data", "knowformer_data"}:
            add(base_path)
            sibling = base_path.parent / (
                "knowformer_data" if base_path.name == "attribute_data" else "attribute_data"
            )
            add(sibling)
        else:
            if subtype:
                add(base_path / subtype)
            add(base_path / "attribute_data")
            add(base_path / "knowformer_data")
            if not variants:
                add(base_path)

        return tuple(variants) if variants else (base_path,)

    def _choose_graph(
        self,
        dataset_name: str,
        kg_number: Optional[int],
        graphs: Iterable[Tuple[str, KnowledgeGraph]],
    ) -> KnowledgeGraph:
        graphs = list(graphs)
        if len(graphs) == 1:
            return graphs[0][1]

        summaries = []
        for label, graph in graphs:
            summary = {"triples": len(graph)}
            summaries.append((label, graph, summary))

        base_summary = summaries[0][2]
        for label, _, summary in summaries[1:]:
            if summary != base_summary:
                logger.warning(
                    "HybEA KG '%s' (kg=%s): mismatch between variants. baseline=%s, %s=%s",
                    dataset_name,
                    kg_number,
                    base_summary,
                    label,
                    summary,
                )

        chosen_label, chosen_graph, chosen_summary = max(
            summaries, key=lambda item: item[2]["triples"]
        )
        logger.info(
            "HybEA KG '%s' (kg=%s): selected variant '%s' with %s",
            dataset_name,
            kg_number,
            chosen_label,
            chosen_summary,
        )
        return chosen_graph

    def _is_attribute_dir(self, data_dir: Path) -> bool:
        if data_dir.name == "attribute_data":
            return True
        if data_dir.name == "knowformer_data":
            return False
        return (data_dir / "ref_pairs").exists()

    def _load_graph(self, data_dir: Path, kg_number: Optional[int]) -> KnowledgeGraph:
        if self._is_attribute_dir(data_dir):
            return self._read_attribute_data(data_dir, kg_number)
        return self._read_knowformer_data(data_dir, kg_number)

    def _read_attribute_data(self, data_dir: Path, kg_number: Optional[int]) -> KnowledgeGraph:
        logger.debug("Parsing attribute-data knowledge graph (kg=%s)", kg_number)
        kg = KnowledgeGraph()

        attr_triple_path = data_dir / f"attr_triples{kg_number}"
        ent_ids_path = data_dir / f"ent_ids_{kg_number}"
        rel_triple_path = data_dir / f"rel_ids_{kg_number}"
        triples_path = data_dir / f"triples_{kg_number}"

        for attr_triple in read_tsv(attr_triple_path):
            if len(attr_triple) < 3:
                logger.debug(
                    "Skipping malformed attribute triple in %s: %s",
                    attr_triple_path,
                    attr_triple,
                )
                continue
            kg.add_attribute_triples(attr_triple)

        ent_ids = {id_: name for id_, name in read_tsv(ent_ids_path)}
        rel_ids = {id_: name for id_, name in read_tsv(rel_triple_path)}

        for triple in read_tsv(triples_path):
            kg.add_relation_triples((ent_ids[triple[0]], rel_ids[triple[1]], ent_ids[triple[2]]))

        return kg

    def _read_knowformer_data(self, data_dir: Path, kg_number: Optional[int]) -> KnowledgeGraph:
        logger.debug("Parsing KnowFormer knowledge graph (kg=%s)", kg_number)
        kg = KnowledgeGraph()

        attr_triple_path = data_dir / f"attr_triples_{kg_number}"
        ent_ids_path = data_dir / f"ent_ids_{kg_number}"
        triples_path = data_dir / ("s_triples.txt" if kg_number == 1 else "t_triples.txt")

        ent_ids = {id_: name for id_, name in read_tsv(ent_ids_path)}

        for attr_triple in read_tsv(attr_triple_path):
            if len(attr_triple) < 3 or attr_triple[0] not in ent_ids:
                logger.debug(
                    "Skipping malformed KnowFormer attribute triple in %s: %s",
                    attr_triple_path,
                    attr_triple,
                )
                continue
            kg.add_attribute_triples((ent_ids[attr_triple[0]], attr_triple[1], attr_triple[2]))

        for triple in read_tsv(triples_path):
            if len(triple) < 3:
                logger.debug(
                    "Skipping malformed KnowFormer relation triple in %s: %s",
                    triples_path,
                    triple,
                )
                continue
            kg.add_relation_triples(triple)

        return kg
