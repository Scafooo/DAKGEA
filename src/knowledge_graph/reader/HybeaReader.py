import os

from src.knowledge_graph.reader.Reader import Reader
from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.logger import get_logger
from src.util.reader import read_tsv

logger = get_logger(__name__)

class HybeaReader(Reader):
    """
    Reader for Hybea files that generates knowledge graphs.

    This class is responsible for reading different types of Hybea files, including
    attribute data and knowformer data. It processes the files, extracts information,
    and constructs a structured KnowledgeGraph instance. This reader is specialized
    for handling Hybea-specific datasets used in knowledge graph construction.

    Attributes:
        file_type: Constant string identifying the file type handled by this
            reader, set to "hybea".
    """
    file_type = "hybea"

    def read(self, dir_path, kg_number = None)  -> KnowledgeGraph:
        logger.info("Reading HybEA knowledge graph from %s", dir_path)
        if "attribute_data" in dir_path:
            return self.read_attribute_data(dir_path, kg_number)
        return self.read_knowformer_data(dir_path, kg_number)

    def read_attribute_data(self, dir_path, kg_number)  -> KnowledgeGraph:
        logger.debug("Parsing attribute-data knowledge graph (kg=%s)", kg_number)
        kg = KnowledgeGraph()

        attr_names_path = os.path.join(dir_path, "attr_names" + str(kg_number))
        attr_triple_path = os.path.join(dir_path, "attr_triples" + str(kg_number))
        ent_ids_path = os.path.join(dir_path, "ent_ids_" + str(kg_number))
        rel_triple_path = os.path.join(dir_path, "rel_ids_" + str(kg_number))
        triples_path = os.path.join(dir_path, "triples_" + str(kg_number))

        for attr_triple in read_tsv(attr_triple_path):
            kg.add_attribute_triples(attr_triple)

        ent_ids = {}
        for id, name in read_tsv(ent_ids_path):
            ent_ids[id] = name

        rel_ids = {}
        for id, name in read_tsv(rel_triple_path):
            rel_ids[id] = name

        for triple in read_tsv(triples_path):
            kg.add_relation_triples((ent_ids[triple[0]],rel_ids[triple[1]],ent_ids[triple[2]]))

        for pair in read_tsv(attr_names_path):
            kg.attr_to_name[pair[0].strip()] = pair[1].strip()

        return kg

    def read_knowformer_data(self, dir_path, kg_number)  -> KnowledgeGraph:
        logger.debug("Parsing KnowFormer knowledge graph (kg=%s)", kg_number)
        kg = KnowledgeGraph()

        attr_names_path = os.path.join(dir_path, "attr_names" + str(kg_number))
        attr_triple_path = os.path.join(dir_path, "attr_triples_" + str(kg_number))
        ent_ids_path = os.path.join(dir_path, "ent_ids_" + str(kg_number))

        triples_path = ""
        if kg_number == 1:
            triples_path = os.path.join(dir_path, "s_triples.txt")
        elif kg_number == 2:
            triples_path = os.path.join(dir_path, "t_triples.txt")

        ent_ids = {}
        for id, name in read_tsv(ent_ids_path):
            ent_ids[id] = name

        for attr_triple in read_tsv(attr_triple_path):
            kg.add_attribute_triples((ent_ids[attr_triple[0]],attr_triple[1],attr_triple[2]))

        for triple in read_tsv(triples_path):
            kg.add_relation_triples(triple)

        for pair in read_tsv(attr_names_path):
            kg.attr_to_name[pair[0].strip()] = pair[1].strip()

        return kg
