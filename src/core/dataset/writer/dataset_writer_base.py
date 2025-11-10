"""Abstract base class for dataset writers."""

from abc import ABC, abstractmethod
from typing import Any

from src.core.dataset.writer.dataset_writer_factory import DatasetWriterFactory


class DatasetWriter(ABC):
    """Base class for dataset writers that register themselves via `file_type`."""

    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            DatasetWriterFactory.register_writer(cls.file_type, cls)

    @abstractmethod
    def write(self, *args: Any, **kwargs: Any):
        """Persist a dataset to storage."""
        raise NotImplementedError
