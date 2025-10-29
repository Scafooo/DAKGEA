"""Factory and auto-discovery utilities for dataset readers."""

import importlib
import pkgutil
from typing import Any, Dict, Type

class ReaderFactory:
    """Registry-backed factory that returns reader implementations by file type."""

    _readers: Dict[str, Type] = {}

    @classmethod
    def register_reader(cls, file_type: str, reader_cls: Type) -> None:
        """Register a reader implementation for a file type key."""
        cls._readers[file_type.lower()] = reader_cls

    @classmethod
    def create_reader(cls, file_type: str, *args: Any, **kwargs: Any):
        """Instantiate a reader, auto-discovering implementations on first use."""
        if not cls._readers:
            cls._autoload("src.dataset.reader")
            cls._autoload("src.knowledge_graph.reader")

        reader_cls = cls._readers.get(file_type.lower())
        if reader_cls is None:
            raise ValueError(f"Unknown file type: {file_type}")
        return reader_cls(*args, **kwargs)

    @classmethod
    def _autoload(cls, package_name: str) -> None:
        """Auto-discover and import all reader subclasses."""
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return
        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, f"{package.__name__}."
        ):
            importlib.import_module(module_name)
