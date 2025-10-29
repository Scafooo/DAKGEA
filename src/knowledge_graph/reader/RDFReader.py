from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.knowledge_graph.reader.Reader import Reader
from src.logger import get_logger

logger = get_logger(__name__, level="DEBUG")

class RDFReader(Reader):
    file_type = "rdf"

    def read(self, file_path) -> KnowledgeGraph:
        logger.debug("Loading RDF graph from %s", file_path)
        kg = KnowledgeGraph()
        kg.parse(file_path)

        return kg
