"""Factory and auto-discovery utilities for dataset writers."""

import importlib
import pkgutil
from typing import Any, Type

from src.util.registry import Registry


class DatasetWriterFactory:
    """Registry-backed factory that returns dataset writer implementations."""

    _registry = Registry("Dataset writer")
    _autoloaded = False

    @classmethod
    def register_writer(cls, file_type: str, writer_cls: Type) -> None:
        """Register a writer implementation for a file type key."""
        cls._registry.add(file_type.lower(), writer_cls)

    @classmethod
    def _ensure_autoload(cls) -> None:
        if cls._autoloaded:
            return
        cls._autoload("src.core.dataset.writer")
        cls._autoloaded = True

    @classmethod
    def create_writer(cls, file_type: str, *args: Any, **kwargs: Any):
        """Instantiate a writer, auto-discovering implementations on first use."""
        cls._ensure_autoload()
        writer_cls = cls._registry.get(file_type.lower())
        return writer_cls(*args, **kwargs)

    @classmethod
    def _autoload(cls, package_name: str) -> None:
        """Auto-discover and import all writer subclasses."""
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return
        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, f"{package.__name__}."
        ):
            importlib.import_module(module_name)
