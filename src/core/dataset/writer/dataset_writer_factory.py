"""Factory utilities for dataset writers."""

import importlib
from typing import Any, Type

from src.util.registry import Registry


class DatasetWriterFactory:
    """Registry-backed factory that returns dataset writer implementations."""

    _registry = Registry("Dataset writer")

    @classmethod
    def register_writer(cls, file_type: str, writer_cls: Type) -> None:
        """Register a writer implementation for a file type key."""
        cls._registry.add(file_type.lower(), writer_cls)

    @classmethod
    def create_writer(cls, file_type: str, *args: Any, **kwargs: Any):
        """Instantiate a writer, importing its module on first use when needed."""
        key = file_type.lower()
        try:
            writer_cls = cls._registry.get(key)
        except ValueError:
            _ensure_registered(key)
            writer_cls = cls._registry.get(key)
        return writer_cls(*args, **kwargs)


_WRITER_MODULES = {
    "openea": "src.core.dataset.writer.openea_dataset_writer",
    "rdf": "src.core.dataset.writer.rdf_dataset_writer",
}


def _ensure_registered(file_type: str) -> None:
    module_path = _WRITER_MODULES.get(file_type)
    if module_path:
        importlib.import_module(module_path)
