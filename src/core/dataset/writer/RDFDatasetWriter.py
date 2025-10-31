from pathlib import Path
from typing import Iterable, Tuple

from src.core.dataset import Dataset
from src.core.dataset.writer.Writer import Writer
from src.core.knowledge_graph.writer import WriterFactory
from src.logger import get_logger
from src.util.writer import write_tsv

logger = get_logger(__name__)


class RDFDatasetWriter(Writer):
    """Persist datasets as RDF N-Triples files with aligned entity mappings."""

    file_type = "rdf"

    def write(self, dataset: Dataset, dir_path: str) -> bool:
        destination = Path(dir_path)
        destination.mkdir(parents=True, exist_ok=True)

        kg_writer = WriterFactory.create_writer("rdf")
        kg_writer.write(dataset.knowledge_graph_source, destination / "graph_source.nt")
        kg_writer.write(dataset.knowledge_graph_target, destination / "graph_target.nt")

        aligned_path = destination / "aligned_entities.tsv"
        aligned_rows: Iterable[Tuple[str, str]] = [
            (str(src), str(tgt)) for src, tgt in dataset.aligned_entities
        ]
        write_tsv(aligned_path, aligned_rows)
        logger.info("Dataset RDF export complete → %s", destination)
        return True
