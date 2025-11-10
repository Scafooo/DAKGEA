"""Abstract base class for dataset readers."""

from abc import ABC, abstractmethod
from typing import Any

from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory


class DatasetReader(ABC):
    """Base class for dataset readers that register themselves via `file_type`."""

    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            DatasetReaderFactory.register_reader(cls.file_type, cls)

    @abstractmethod
    def read(self, *args: Any, **kwargs: Any):
        """Read data from storage and return a dataset instance."""
        raise NotImplementedError
