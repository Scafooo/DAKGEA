"""Abstract base class for dataset writers."""

from abc import ABC, abstractmethod
from typing import Any

from src.dataset.writer.WriterFactory import WriterFactory


class Writer(ABC):
    """Base class for dataset writers that register themselves via `file_type`."""

    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            WriterFactory.register_writer(cls.file_type, cls)

    @abstractmethod
    def write(self, *args: Any, **kwargs: Any) -> bool:
        """Persist a dataset to disk."""
        raise NotImplementedError
