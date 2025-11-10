"""Factory utilities for dataset readers."""

import importlib
from typing import Any, Type

from src.util.registry import Registry


class DatasetReaderFactory:
    """Registry-backed factory that returns dataset reader implementations by file type."""

    _registry = Registry("Dataset reader")

    @classmethod
    def register_reader(cls, file_type: str, reader_cls: Type) -> None:
        """Register a reader implementation for a file type key."""
        cls._registry.add(file_type.lower(), reader_cls)

    @classmethod
    def create_reader(cls, file_type: str, *args: Any, **kwargs: Any):
        """Instantiate a reader that has been previously registered."""
        key = file_type.lower()
        try:
            reader_cls = cls._registry.get(key)
        except ValueError:
            _ensure_registered(key)
            reader_cls = cls._registry.get(key)
        return reader_cls(*args, **kwargs)


_READER_MODULES = {
    "hybea": "src.core.dataset.reader.hybea_dataset_reader",
    "rdf": "src.core.dataset.reader.rdf_dataset_reader",
}


def _ensure_registered(file_type: str) -> None:
    module_path = _READER_MODULES.get(file_type.lower())
    if module_path:
        importlib.import_module(module_path)
