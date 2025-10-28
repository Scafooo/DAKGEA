from src.logger import logger
from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.knowledge_graph.writer.Writer import Writer


class RDFWriter(Writer):

    file_type = "rdf"

    def write(self, dir_path, kg : KnowledgeGraph, kg_number = None) -> bool:

        logger.info("Knowledge Graph RDF Export Start")

        kg.serialize(destination=dir_path)

        logger.info("Knowledge Graph RDF Export End")

        return True