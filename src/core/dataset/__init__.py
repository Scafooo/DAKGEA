"""Core dataset module aggregating data structures and IO helpers."""

from src.core.dataset.dataset import Dataset
from src.core.dataset.reader import Reader, ReaderFactory
from src.core.dataset.writer import Writer, WriterFactory

__all__ = [
    "Dataset",
    "Reader",
    "ReaderFactory",
    "Writer",
    "WriterFactory",
]
