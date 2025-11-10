"""Core domain abstractions shared across DAKGEA modules."""

from src.core.dataset import (
    Dataset,
    DatasetReader,
    DatasetReaderFactory,
    DatasetWriter,
    DatasetWriterFactory,
)
from src.core.knowledge_graph import KnowledgeGraph
from src.core.knowledge_graph import (
    KnowledgeGraphReader,
    KnowledgeGraphReaderFactory,
    KnowledgeGraphWriter,
    KnowledgeGraphWriterFactory,
)

__all__ = [
    "Dataset",
    "DatasetReader",
    "DatasetReaderFactory",
    "DatasetWriter",
    "DatasetWriterFactory",
    "KnowledgeGraph",
    "KnowledgeGraphReader",
    "KnowledgeGraphReaderFactory",
    "KnowledgeGraphWriter",
    "KnowledgeGraphWriterFactory",
]
