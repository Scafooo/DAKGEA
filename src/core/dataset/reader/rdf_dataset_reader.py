from pathlib import Path
from typing import Optional

from src.core.dataset import Dataset
from src.core.dataset.reader.dataset_reader_base import DatasetReader
from src.core.knowledge_graph.reader import KnowledgeGraphReaderFactory
from src.utils.reader import read_tsv


class RDFDatasetReader(DatasetReader):
    """Reconstruct datasets exported in RDF format (TTL or N-Triples)."""

    file_type = "rdf"

    def read(self, dir_path: str, **_) -> Dataset:
        base_path = Path(dir_path)
        kg_reader = KnowledgeGraphReaderFactory.create_reader("rdf")

        kg_source_path = self._resolve_graph_path(base_path, "graph_source")
        kg_target_path = self._resolve_graph_path(base_path, "graph_target")

        kg_source = kg_reader.read(str(kg_source_path))
        kg_target = kg_reader.read(str(kg_target_path))

        aligned_path = base_path / "aligned_entities.tsv"
        aligned_entities = [(row[0], row[1]) for row in read_tsv(aligned_path)]

        return Dataset(kg_source, kg_target, aligned_entities)

    @staticmethod
    def _resolve_graph_path(base_path: Path, stem: str) -> Path:
        """Return the first existing RDF graph file matching known suffixes."""
        for suffix in (".nt", ".ttl"):
            candidate = base_path / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Unable to locate RDF graph file for '{stem}' under {base_path}"
        )
