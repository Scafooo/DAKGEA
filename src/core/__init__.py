"""Core domain abstractions shared across DAKGEA modules."""

from src.core.dataset import Dataset, Reader as DatasetReader, ReaderFactory as DatasetReaderFactory
from src.core.dataset import Writer as DatasetWriter, WriterFactory as DatasetWriterFactory
from src.core.knowledge_graph import KnowledgeGraph
from src.core.knowledge_graph import (
    Reader as KnowledgeGraphReader,
    ReaderFactory as KnowledgeGraphReaderFactory,
    Writer as KnowledgeGraphWriter,
    WriterFactory as KnowledgeGraphWriterFactory,
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
