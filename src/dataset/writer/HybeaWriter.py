"""Writer for HybEA-formatted datasets."""

import copy
import os
import unicodedata
from typing import Iterable, Tuple

from rdflib.term import URIRef

from src.dataset.Dataset import Dataset
from src.dataset.writer.Writer import Writer
from src.knowledge_graph.writer.WriterFactory import WriterFactory
from src.logger import get_logger
from src.util.reader import read_tsv
from src.util.writer import write_tsv

logger = get_logger(__name__)

class HybeaWriter(Writer):
    """Persist datasets back to HybEA/KnowFormer directory layouts."""

    file_type = "hybea"

    def write(self, dataset: Dataset, dir_path: str) -> bool:
        """Export a dataset to the requested HybEA directory layout."""

        logger.info("Dataset Hybea Export Start")

        kg_writer_1 = WriterFactory.create_writer(self.file_type)
        kg_writer_1.write(dataset.knowledge_graph_source, dir_path, kg_number=1)

        kg_writer_2 = WriterFactory.create_writer(self.file_type)
        kg_writer_2.write(dataset.knowledge_graph_target, dir_path, kg_number=2)

        if "attribute_data" in dir_path:
            return self._write_aligned_entities_attribute(dataset.aligned_entities, dir_path)
        return self._write_aligned_entities_knowformer(dataset, dir_path)

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
        n = len(aligned_list)
        n1 = int(n * 0.7)
        n2 = int(n * 0.9)
        logger.debug(
            "Preparing %d aligned entity pairs for attribute export (%d/%d/%d splits)",
            n,
            n1,
            n2 - n1,
            n - n2,
        )

        def norm_entity(entity):
            """Normalize aligned entity entries (URIRef or str) to NFC strings."""
            return unicodedata.normalize("NFC", str(entity))

        list_aligned_entities = sorted(
            [(norm_entity(e1), norm_entity(e2)) for e1, e2 in aligned_list]
        )

        missing = []
        for e1, e2 in list_aligned_entities:
            if e1 not in ent_ids or e2 not in ent_ids:
                missing.append((e1, e2))
        if missing:
            sample = ", ".join([f"({e1}, {e2})" for e1, e2 in missing])
            raise KeyError(f"Some aligned entities were not found in ent_ids mapping. "
                           f"Missing examples: {sample}. "
                           f"Ensure ent_ids_1/ent_ids_2 contain all aligned entities after normalization.")

        ref_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[:n1]]
        sup_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[n1:n2]]
        valid_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[n2:]]

        write_tsv(os.path.join(dir_path, "ref_pairs"), ref_pairs)
        write_tsv(os.path.join(dir_path, "sup_pairs"), sup_pairs)
        write_tsv(os.path.join(dir_path, "valid_pairs"), valid_pairs)

        logger.info("Dataset Hybea Export End")

        return True


    def _write_aligned_entities_knowformer(self, dataset: Dataset, dir_path: str) -> bool:
        """Persist aligned entities for the KnowFormer layout."""

        ENT_ILLS = os.path.join(dir_path, "ent_ILLs.txt")
        REF_ENTS = os.path.join(dir_path, "ref_ents.txt")
        SUP_ENTS = os.path.join(dir_path, "sup_ents.txt")
        VALID_ENTS = os.path.join(dir_path, "valid_ents.txt")
        TRAIN_TRIPLES = os.path.join(dir_path, "train.triples.txt")
        VOCAB = os.path.join(dir_path, "vocab.txt")

        def norm_entity(entity):
            """Normalize aligned entity entries (URIRef or str) to NFC strings."""
            return unicodedata.normalize("NFC", str(entity))

        list_aligned_entities = sorted(
            (norm_entity(e1), norm_entity(e2)) for e1, e2 in dataset.aligned_entities
        )

        n = len(list_aligned_entities)
        n1 = int(n * 0.7)
        n2 = int(n * 0.9)

        ent_ILLs_pairs = list_aligned_entities
        ref_pairs = list_aligned_entities[:n1]
        sup_pairs = list_aligned_entities[n1:n2]
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

        train_triples = set()

        for subj, pred, obj in dataset.knowledge_graph_source:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                train_triples.add((str(subj), str(pred), str(obj)))

        for subj, pred, obj in dataset.knowledge_graph_target:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                train_triples.add((str(subj), str(pred), str(obj)))

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

        train_triples_expanded = copy.deepcopy(train_triples)

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

        train_triples_expanded_list = list(train_triples_expanded)

        train_triples_final = []

        for triple in train_triples_expanded_list:
            train_triples_final.append((triple[0], triple[1], triple[2], "MASK_0"))
            train_triples_final.append((triple[0], triple[1], triple[2], "MASK_2"))

        write_tsv(ENT_ILLS, ent_ILLs_pairs)
        write_tsv(REF_ENTS, ref_pairs)
        write_tsv(SUP_ENTS, sup_pairs)
        write_tsv(VALID_ENTS, valid_pairs)
        write_tsv(TRAIN_TRIPLES, train_triples_final)
        write_tsv(VOCAB, list(vocab.keys()))

        logger.info("Dataset Hybea Export End")

        return True
