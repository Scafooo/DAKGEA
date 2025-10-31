from pathlib import Path

from src.core.knowledge_graph import KnowledgeGraph
from src.core.knowledge_graph.reader.Reader import Reader
from src.logger import get_logger

logger = get_logger(__name__, level="DEBUG")

class RDFReader(Reader):
    file_type = "rdf"

    def read(self, file_path, **_) -> KnowledgeGraph:
        logger.debug("Loading RDF graph from %s", file_path)
        kg = KnowledgeGraph()
        format_hint = self._infer_format(file_path)
        if format_hint:
            kg.parse(file_path, format=format_hint)
        else:
            kg.parse(file_path)

        return kg

    @staticmethod
    def _infer_format(file_path) -> str:
        """Infer rdflib parse format from the file extension when possible."""
        suffix = Path(file_path).suffix.lower()
        if suffix in {".nt", ".ntriples"}:
            return "nt"
        if suffix in {".ttl", ".turtle"}:
            return "turtle"
        return ""
