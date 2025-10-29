"""Abstract base class for dataset readers."""

from abc import ABC, abstractmethod
from typing import Any

from src.dataset.Dataset import Dataset
from src.dataset.reader.ReaderFactory import ReaderFactory


class Reader(ABC):
    """Base class for dataset readers that register themselves via `file_type`."""

    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            ReaderFactory.register_reader(cls.file_type, cls)

    @abstractmethod
    def read(self, *args: Any, **kwargs: Any) -> Dataset:
        """Read data from storage and return a dataset instance."""
        raise NotImplementedError
