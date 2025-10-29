from pathlib import Path

from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph
from src.knowledge_graph.writer.Writer import Writer
from src.logger import get_logger

logger = get_logger(__name__)


class RDFWriter(Writer):

    file_type = "rdf"

    def write(self, kg: KnowledgeGraph, output_path, kg_number=None) -> bool:
        logger.info("Knowledge Graph RDF Export Start")

        destination = Path(output_path)
        if destination.is_dir() or destination.suffix == "":
            base_name = "kg"
            if kg_number is not None:
                base_name = f"kg_{kg_number}"
            destination = destination / f"{base_name}.nt"

        destination.parent.mkdir(parents=True, exist_ok=True)
        kg.serialize(destination=str(destination), format="nt")

        logger.info("Knowledge Graph RDF Export End")

        return True
