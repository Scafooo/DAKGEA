from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.knowledge_graph.reader.Reader import Reader
from src.logger import logger

class RDFReader(Reader):
    file_type = "rdf"

    def read(self, file_path) -> KnowledgeGraph:
        logger.info(f"Reading file: {file_path}")
        kg = KnowledgeGraph()
        kg.parse(file_path)

        return kg