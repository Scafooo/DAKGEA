"""Factory and auto-discovery utilities for dataset readers."""

import importlib
import pkgutil
from typing import Any, Type

from src.util.registry import Registry


class ReaderFactory:
    """Registry-backed factory that returns reader implementations by file type."""

    _registry = Registry("Dataset reader")
    _autoloaded = False

    @classmethod
    def register_reader(cls, file_type: str, reader_cls: Type) -> None:
        """Register a reader implementation for a file type key."""
        cls._registry.add(file_type.lower(), reader_cls)

    @classmethod
    def _ensure_autoload(cls) -> None:
        if cls._autoloaded:
            return
        cls._autoload("src.core.dataset.reader")
        cls._autoload("src.core.knowledge_graph.reader")
        cls._autoloaded = True

    @classmethod
    def create_reader(cls, file_type: str, *args: Any, **kwargs: Any):
        """Instantiate a reader, auto-discovering implementations on first use."""
        cls._ensure_autoload()
        reader_cls = cls._registry.get(file_type.lower())
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
