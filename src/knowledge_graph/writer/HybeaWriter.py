import os
from rdflib import Literal, URIRef
from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.knowledge_graph.writer.Writer import Writer
from src.logger import logger
from src.util.reader import read_tsv
from src.util.writer import write_tsv


class HybeaWriter(Writer):
    file_type = "hybea"

    def write(self, dir_path, kg : KnowledgeGraph, kg_number = None) -> bool:

        logger.info("Knowledge Graph Hybea Export Start")

        if "attribute_data" in dir_path:
            return self.write_attribute(dir_path, kg, kg_number)
        else: #knowformer
            return self.write_knowformer(dir_path, kg, kg_number)



    def write_attribute(self, dir_path, kg, kg_number) -> bool:

        logger.info("Knowledge Graph Hybea Attribute")

        ATTR_NAMES = os.path.join(dir_path, "attr_names" + str(kg_number))
        ATTR_TRIPLE = os.path.join(dir_path, "attr_triples" + str(kg_number))
        ENT_IDS = os.path.join(dir_path, "ent_ids_" + str(kg_number))
        REL_IDS = os.path.join(dir_path, "rel_ids_" + str(kg_number))
        TRIPLES = os.path.join(dir_path, "triples_" + str(kg_number))

        attr_triples = []
        e_id = 0
        r_id = 0
        ent_ids = {}
        rel_ids = {}
        triples = []

        if kg_number == 2:
            ENT_IDS_1 = os.path.join(dir_path, "ent_ids_1")
            e_id = int(read_tsv(ENT_IDS_1)[-1][0]) + 1
            REL_IDS_1 = os.path.join(dir_path, "rel_ids_1")
            r_id = int(read_tsv(REL_IDS_1)[-1][0]) + 1

        ordered_kg_triples = sorted([[s, p, o] for s, p, o in kg])

        for s, p, o in ordered_kg_triples:
            # attribute triple
            if isinstance(o, Literal):
                # print("1","s: ",s,"p: ",p,"o: ",o)
                attr_triples.append((str(s), str(p), str(o)))
                if str(s) not in ent_ids:
                    ent_ids[str(s)] = e_id
                    e_id += 1
            # relation triple
            else:
                if str(s) not in ent_ids:
                    ent_ids[str(s)] = e_id
                    e_id += 1
                if str(p) not in rel_ids:
                    rel_ids[str(p)] = r_id
                    r_id += 1
                if str(o) not in ent_ids:
                    ent_ids[str(o)] = e_id
                    e_id += 1

                triples.append((str(ent_ids[str(s)]), str(rel_ids[str(p)]), str(ent_ids[str(o)])))

        write_tsv(ATTR_NAMES,list(kg.attr_to_name.items()))
        write_tsv(ATTR_TRIPLE, attr_triples)
        write_tsv(ENT_IDS, [[str(v), str(k)] for v, k in sorted([[v, k] for k, v in ent_ids.items()], key=lambda x: x[0])])
        write_tsv(REL_IDS, [[str(v), str(k)] for v, k in sorted([[v, k] for k, v in rel_ids.items()], key=lambda x: x[0])])
        write_tsv(TRIPLES, triples)

        logger.info("Knowledge Graph Hybea Export End")

        return True


    def write_knowformer(self, dir_path, kg, kg_number) -> bool:

        logger.info("Knowledge Graph Hybea Knowformer Export")

        ATTR_NAMES = os.path.join(dir_path, "attr_names" + str(kg_number))
        ATTR_TRIPLE = os.path.join(dir_path, "attr_triples_" + str(kg_number))
        ENT_IDS = os.path.join(dir_path, "ent_ids_" + str(kg_number))
        TRIPLES = None
        if kg_number == 1:
            TRIPLES = os.path.join(dir_path, "s_triples.txt")
        else:
            TRIPLES = os.path.join(dir_path, "t_triples.txt")

        ent_ids = {}
        attr_triples = []
        triples = []
        e_id = 0

        if kg_number == 2:
            ENT_IDS_1 = os.path.join(dir_path, "ent_ids_1")
            e_id = int(read_tsv(ENT_IDS_1)[-1][0]) + 1

        ordered_kg_triples = sorted([[s, p, o] for s, p, o in kg])

        for s, p, o in ordered_kg_triples:
            if isinstance(s, URIRef):
                if str(s) not in ent_ids:
                    ent_ids[str(s)] = e_id
                    e_id += 1
            if isinstance(o, URIRef):
                if str(o) not in ent_ids:
                    ent_ids[str(o)] = e_id
                    e_id += 1

        for s, p, o in ordered_kg_triples:
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                triples.append((str(s), str(p), str(o)))
            if isinstance(o, Literal):
                attr_triples.append((str(ent_ids[str(s)]), str(p), str(o)))

        write_tsv(ATTR_NAMES, list(kg.attr_to_name.items()))
        write_tsv(ATTR_TRIPLE, attr_triples)
        write_tsv(ENT_IDS,
                  [[str(v), str(k)] for v, k in sorted([[v, k] for k, v in ent_ids.items()], key=lambda x: x[0])])
        write_tsv(TRIPLES, triples)

        logger.info("Knowledge Graph Hybea Export End")

        return True