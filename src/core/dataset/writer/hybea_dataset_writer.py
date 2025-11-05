"""Writer for HybEA-formatted datasets."""

import copy
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Iterable, Tuple, Optional

from rdflib.term import URIRef

from src.core.dataset import Dataset
from src.core.dataset.writer import DatasetWriter
from src.core.knowledge_graph.writer import KnowledgeGraphWriterFactory
from src.logger import get_logger, get_structured_logger
from src.util.reader import read_tsv
from src.util.writer import write_tsv

DEFAULT_DATASET_NAME = "dataset"

logger = get_logger(__name__)
slogger = get_structured_logger(__name__)

class HybeaDatasetWriter(DatasetWriter):
    """Persist datasets back to HybEA/KnowFormer directory layouts."""

    file_type = "hybea"

    def write(self, dataset: Dataset, dir_path: str, *, dataset_name: Optional[str] = None) -> bool:
        """Export a dataset to the requested HybEA directory layout."""

        slogger.section("HybEA Dataset Export")

        base_dir = Path(dir_path)
        if base_dir.name in {"attribute_data", "knowformer_data"}:
            targets = [base_dir]
        else:
            targets = [base_dir / "attribute_data", base_dir / "knowformer_data"]

        if dataset_name is None:
            dataset_name = DEFAULT_DATASET_NAME

        slogger.table("Export Configuration", {
            "Dataset Name": dataset_name,
            "Output Targets": len(targets),
            "Base Directory": str(base_dir)
        })

        for idx, target in enumerate(targets, 1):
            slogger.subsection(f"Processing Target {idx}/{len(targets)}: {target.name}")

            target.mkdir(parents=True, exist_ok=True)
            if target.name == "knowformer_data":
                destination = target
            else:
                destination = target

            logger.info(f"Writing KG1 (Source) to {destination}")
            kg_writer_1 = KnowledgeGraphWriterFactory.create_writer(self.file_type)
            kg_writer_1.write(dataset.knowledge_graph_source, str(destination), kg_number=1)

            logger.info(f"Writing KG2 (Target) to {destination}")
            kg_writer_2 = KnowledgeGraphWriterFactory.create_writer(self.file_type)
            kg_writer_2.write(dataset.knowledge_graph_target, str(destination), kg_number=2)

            if target.name == "attribute_data":
                logger.info("Writing aligned entities (attribute layout)")
                self._write_aligned_entities_attribute(dataset.aligned_entities, str(destination))
            else:
                logger.info("Writing aligned entities (KnowFormer layout)")
                self._write_aligned_entities_knowformer(dataset, str(destination))
                if destination != target:
                    logger.info("Mirroring structure files")
                    self._mirror_structure_files(destination, target)

        slogger.success("Dataset HybEA export completed successfully")
        return True

    def _write_aligned_entities_attribute(
        self, aligned_entities: Iterable[Tuple[URIRef, URIRef]], dir_path: str
    ) -> bool:
        """Persist aligned entities for the attribute-data layout."""

        ent_ids_1 = read_tsv(os.path.join(dir_path, "ent_ids_1"))
        ent_ids_2 = read_tsv(os.path.join(dir_path, "ent_ids_2"))
        ent_ids = {}

        for elem in ent_ids_1:
            key = unicodedata.normalize("NFC", str(elem[1]))
            ent_ids[key] = str(elem[0])
        for elem in ent_ids_2:
            key = unicodedata.normalize("NFC", str(elem[1]))
            ent_ids[key] = str(elem[0])

        aligned_list = list(aligned_entities)
        original_pairs = len(aligned_list)

        def norm_entity(entity):
            """Normalize aligned entity entries (URIRef or str) to NFC strings."""
            return unicodedata.normalize("NFC", str(entity))

        normalised_pairs = sorted(
            [(norm_entity(e1), norm_entity(e2)) for e1, e2 in aligned_list]
        )

        missing = []
        list_aligned_entities = []
        for e1, e2 in normalised_pairs:
            if e1 not in ent_ids or e2 not in ent_ids:
                missing.append((e1, e2))
                continue
            list_aligned_entities.append((e1, e2))

        if missing:
            sample = ", ".join([f"({e1}, {e2})" for e1, e2 in missing[:5]])
            logger.warning(
                "Skipping %d aligned pairs missing from ent_ids mapping. Samples: %s",
                len(missing),
                sample,
            )
            if len(missing) > 5:
                logger.warning("Additional %d missing pairs omitted from log.", len(missing) - 5)

        n = len(list_aligned_entities)
        n1 = int(n * 0.9)
        n2 = int(n * 0.7)
        logger.debug(
            "Preparing %d aligned entity pairs for attribute export (train=%d/test=%d/valid=%d). "
            "Dropped %d missing pairs from original %d.",
            n,
            n1,
            n2 - n1,
            n - n2,
            len(missing),
            original_pairs,
        )

        # FIXED: Inverted to have train=20% (large) and test=20% (small)
        # sup_pairs is used as train_ill, ref_pairs is used as test_ill
        sup_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[:n1]]
        ref_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[n1:n2]]
        valid_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[n2:]]

        write_tsv(os.path.join(dir_path, "ref_pairs"), ref_pairs)
        write_tsv(os.path.join(dir_path, "sup_pairs"), sup_pairs)
        write_tsv(os.path.join(dir_path, "valid_pairs"), valid_pairs)

        return True


    def _write_aligned_entities_knowformer(self, dataset: Dataset, dir_path: str) -> bool:
        """Persist aligned entities for the KnowFormer layout."""

        slogger.subsection("KnowFormer Aligned Entities Export")

        ENT_ILLS = os.path.join(dir_path, "ent_ILLs.txt")
        REF_ENTS = os.path.join(dir_path, "ref_ents.txt")
        SUP_ENTS = os.path.join(dir_path, "sup_ents.txt")
        VALID_ENTS = os.path.join(dir_path, "valid_ents.txt")
        TRAIN_TRIPLES = os.path.join(dir_path, "train.triples.txt")
        VOCAB = os.path.join(dir_path, "vocab.txt")

        def norm_entity(entity):
            """Normalize aligned entity entries (URIRef or str) to NFC strings."""
            return unicodedata.normalize("NFC", str(entity))

        logger.info("Normalizing aligned entities...")
        list_aligned_entities = sorted(
            (norm_entity(e1), norm_entity(e2)) for e1, e2 in dataset.aligned_entities
        )

        n = len(list_aligned_entities)
        n1 = int(n * 0.7)
        n2 = int(n * 0.9)

        slogger.table("Entity Split Configuration", {
            "Total Aligned Entities": n,
            "Support Set (70%) [train]": n1,
            "Reference Set (20%) [test]": n2 - n1,
            "Validation Set (10%)": n - n2
        })

        ent_ILLs_pairs = list_aligned_entities
        # FIXED: Inverted to have train=70% (large) and test=20% (small)
        sup_pairs = list_aligned_entities[:n1]
        ref_pairs = list_aligned_entities[n1:n2]
        valid_pairs = list_aligned_entities[n2:]
        vocab = {}

        vocab["[PAD]"] = None

        for i in range(95):
            vocab[f"[unused{i}]"] = None

        vocab["[UNK]"] = None
        vocab["[CLS]"] = None
        vocab["[SEP]"] = None
        vocab["[MASK]"] = None

        vocab_ent_1 = {}
        vocab_rel_1 = {}
        vocab_ent_2 = {}
        vocab_rel_2 = {}

        for subj, pred, obj in dataset.knowledge_graph_source:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                normalized_subj = unicodedata.normalize("NFC", str(subj))
                normalized_obj = unicodedata.normalize("NFC", str(obj))
                vocab_ent_1[normalized_subj] = None
                vocab_ent_1[normalized_obj] = None
                vocab_rel_1[pred] = None

        for subj, pred, obj in dataset.knowledge_graph_target:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                normalized_subj = unicodedata.normalize("NFC", str(subj))
                normalized_obj = unicodedata.normalize("NFC", str(obj))
                vocab_ent_2[normalized_subj] = None
                vocab_ent_2[normalized_obj] = None
                vocab_rel_2[pred] = None

        for e1, e2 in ref_pairs:
            vocab_ent_1[e1] = None
            vocab_ent_2[e2] = None

        for e1, e2 in valid_pairs:
            vocab_ent_1[e1] = None
            vocab_ent_2[e2] = None

        for elem in vocab_ent_1:
            vocab[elem] = None
        for elem in vocab_ent_2:
            vocab[elem] = None
        for elem in vocab_rel_1:
            vocab[elem] = None
        for elem in vocab_rel_2:
            vocab[elem] = None

        # Collect source triples (from KG1)
        logger.info("Collecting source triples (KG1)...")
        source_triples = set()
        for subj, pred, obj in dataset.knowledge_graph_source:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                source_triples.add((str(subj), str(pred), str(obj)))
        logger.debug(f"Collected {len(source_triples)} source triples")

        # Collect target triples (from KG2)
        logger.info("Collecting target triples (KG2)...")
        target_triples = set()
        for subj, pred, obj in dataset.knowledge_graph_target:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                target_triples.add((str(subj), str(pred), str(obj)))
        logger.debug(f"Collected {len(target_triples)} target triples")

        # Create mapping from target entities to source entities using aligned entities
        logger.info("Creating entity mapping from target to source...")
        target_to_source_map = {}
        for e1, e2 in ent_ILLs_pairs:
            # e1 is from source (DBpedia), e2 is from target (Wikidata)
            target_to_source_map[e2] = e1
        logger.debug(f"Created mapping for {len(target_to_source_map)} entities")

        # Merge triples: start with source triples, then add target triples with mapped entities
        logger.info("Merging triples with entity mapping...")
        train_triples = copy.deepcopy(source_triples)

        for subj, pred, obj in target_triples:
            # Map target entities to source entities
            mapped_subj = target_to_source_map.get(subj, subj)
            mapped_obj = target_to_source_map.get(obj, obj)
            train_triples.add((mapped_subj, pred, mapped_obj))
        logger.debug(f"Merged triples: {len(train_triples)} total")

        # Expand triples using equivalence classes from ref_pairs and valid_pairs
        logger.info("Building equivalence classes from aligned entities...")
        res_valid_set = set(ref_pairs + valid_pairs)

        eq_classes = {}
        for e1, e2 in res_valid_set:
            if str(e1) not in eq_classes:
                eq_classes[str(e1)] = [str(e2)]
            else:
                eq_classes[str(e1)].append(str(e2))
            if str(e2) not in eq_classes:
                eq_classes[str(e2)] = [str(e1)]
            else:
                eq_classes[str(e2)].append(str(e1))
        logger.debug(f"Built {len(eq_classes)} equivalence classes")

        logger.info("Expanding training triples using equivalence classes...")
        train_triples_expanded = copy.deepcopy(train_triples)
        initial_count = len(train_triples)

        for train_triple in train_triples:
            if train_triple[0] in eq_classes:
                for eq_class in eq_classes[train_triple[0]]:
                    train_triples_expanded.add(
                        (eq_class, train_triple[1], train_triple[2])
                    )
            if train_triple[2] in eq_classes:
                for eq_class in eq_classes[train_triple[2]]:
                    train_triples_expanded.add(
                        (train_triple[0], train_triple[1], eq_class)
                    )

        expansion_ratio = len(train_triples_expanded) / initial_count if initial_count > 0 else 1
        logger.info(f"Triple expansion: {initial_count} → {len(train_triples_expanded)} (ratio: {expansion_ratio:.2f}x)")

        train_triples_expanded_list = list(train_triples_expanded)

        # Write entity alignment files
        write_tsv(ENT_ILLS, ent_ILLs_pairs)
        write_tsv(REF_ENTS, ref_pairs)
        write_tsv(SUP_ENTS, sup_pairs)
        write_tsv(VALID_ENTS, valid_pairs)

        # Write training triples with MASK tokens using the helper function
        self._write_triples_with_masks(TRAIN_TRIPLES, train_triples_expanded_list)
        logger.info(f"✓ Wrote {len(train_triples_expanded_list) * 2} triple instances (with MASK tokens)")

        # Write vocab.txt with entities and relations from training triples
        self._write_vocab_from_triples(VOCAB, train_triples_expanded_list)

        slogger.success("All output files written successfully")

        logger.info("Dataset Hybea Export End")

        return True

    def _write_triples_with_masks(self, triples_path: str, triples: list) -> None:
        """Write triples with MASK tokens for KnowFormer compatibility.

        Each triple is written twice: once with MASK_0 (masking head entity)
        and once with MASK_2 (masking tail entity).
        """
        with open(triples_path, "w", encoding="utf-8") as f:
            for triple in triples:
                # Convert triple to list of strings
                triple_str = [str(item) for item in triple]
                triple_line = "\t".join(triple_str)
                # Write with MASK_0 (mask head entity)
                f.write(triple_line + "\tMASK_0\n")
                # Write with MASK_2 (mask tail entity)
                f.write(triple_line + "\tMASK_2\n")

    def _write_vocab_from_triples(self, vocab_path: str, triples: list) -> None:
        """Write vocabulary file from training triples.

        Extracts entities and relations from triples and writes them to vocab file
        in the format expected by KnowFormer (with special tokens at the beginning).
        """
        entities_set = set()
        relations_set = set()

        # Extract entities and relations from triples
        for triple in triples:
            # triple is (subject, predicate, object)
            entities_set.add(str(triple[0]))
            relations_set.add(str(triple[1]))
            entities_set.add(str(triple[2]))

        # Sort for consistency
        entities_list = sorted(list(entities_set))
        relations_list = sorted(list(relations_set))

        # Write vocab file with special tokens first
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("[PAD]\n")
            for i in range(0, 95):
                f.write("[unused{}]\n".format(i))
            f.write("[UNK]\n")
            f.write("[CLS]\n")
            f.write("[SEP]\n")
            f.write("[MASK]\n")
            for entity in entities_list:
                f.write(entity + "\n")
            for relation in relations_list:
                f.write(relation + "\n")

    def _mirror_structure_files(self, source_dir: Path, target_dir: Path) -> None:
        """Duplicate core structure artifacts at the parent level for legacy lookups."""

        filenames = {
            "vocab.txt",
            "ent_ids_1",
            "ent_ids_2",
            "ent_ILLs.txt",
            "ref_ents.txt",
            "sup_ents.txt",
            "valid_ents.txt",
            "train.triples.txt",
        }

        for name in filenames:
            src = source_dir / name
            dst = target_dir / name
            if src.exists():
                shutil.copy2(src, dst)
