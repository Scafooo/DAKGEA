import os
from typing import Dict, Iterable, Set, Tuple
import unicodedata

from rdflib import URIRef

from src.knowledge_graph.reader.ReaderFactory import ReaderFactory
from src.util.reader import read_tsv
from src.dataset.reader.Reader import Reader
from src.dataset.Dataset import Dataset


class HybeaReader(Reader):
    file_type = "hybea"

    ATTR_ENT_IDS_1 = "ent_ids_1"
    ATTR_ENT_IDS_2 = "ent_ids_2"
    ATTR_REF_PAIRS = "ref_pairs"
    ATTR_SUP_PAIRS = "sup_pairs"
    ATTR_VALID_PAIRS = "valid_pairs"

    KF_ILL = "ent_ILLs.txt"

    def read(self, dir_path: str) -> Dataset:

        kg1_reader = ReaderFactory.create_reader(self.file_type)
        kg1 = kg1_reader.read(dir_path, kg_number=1)

        kg2_reader = ReaderFactory.create_reader(self.file_type)
        kg2 = kg2_reader.read(dir_path, kg_number=2)

        if "attribute_data" in dir_path:
            aligned_entities = self._aligned_entities_attribute_data(dir_path)
        else:  # knowformer_data
            aligned_entities = self._aligned_entities_knowformer_data(dir_path)

        return Dataset(kg1, kg2, aligned_entities)

    def _load_entity_ids(self, ent_ids_1_path: str, ent_ids_2_path: str) -> Dict[str, str]:
        ent_ids: Dict[str, str] = {}
        for rows in (read_tsv(ent_ids_1_path), read_tsv(ent_ids_2_path)):
            for src, dst in rows:
                # normalize to avoid mismatches on composed/ decomposed unicode
                nsrc = unicodedata.normalize("NFC", src)
                ndst = unicodedata.normalize("NFC", dst)
                ent_ids[nsrc] = ndst
        return ent_ids

    def _collect_aligned_pairs(self, ent_ids: Dict[str, str], pair_files: Iterable[str]) -> Set[Tuple[URIRef, URIRef]]:

        aligned: Set[Tuple[URIRef, URIRef]] = set()
        missing = 0
        for file_path in pair_files:
            for left, right in read_tsv(file_path):
                nleft = unicodedata.normalize("NFC", left)
                nright = unicodedata.normalize("NFC", right)
                if nleft in ent_ids and nright in ent_ids:
                    aligned.add((URIRef(ent_ids[nleft]), URIRef(ent_ids[nright])))
                else:
                    missing += 1

        return aligned

    def _aligned_entities_attribute_data(self, dir_path: str) -> Set[Tuple[URIRef, URIRef]]:
        ent_ids_1_path = os.path.join(dir_path, self.ATTR_ENT_IDS_1)
        ent_ids_2_path = os.path.join(dir_path, self.ATTR_ENT_IDS_2)
        ent_ids = self._load_entity_ids(ent_ids_1_path, ent_ids_2_path)

        pair_files = (
            os.path.join(dir_path, self.ATTR_REF_PAIRS),
            os.path.join(dir_path, self.ATTR_SUP_PAIRS),
            os.path.join(dir_path, self.ATTR_VALID_PAIRS),
        )
        return self._collect_aligned_pairs(ent_ids, pair_files)

    def _aligned_entities_knowformer_data(self, dir_path: str) -> Set[Tuple[URIRef, URIRef]]:

        pair_files = (
            os.path.join(dir_path, self.KF_ILL),
        )

        aligned: Set[Tuple[URIRef, URIRef]] = set()
        for file_path in pair_files:
            for left, right in read_tsv(file_path):
                nleft = unicodedata.normalize("NFC", left)
                nright = unicodedata.normalize("NFC", right)
                aligned.add((URIRef(nleft), URIRef(nright)))

        return aligned