import importlib
import pkgutil
from typing import Any, Type

from src.util.registry import Registry


class KnowledgeGraphWriterFactory:
    _registry = Registry("Knowledge graph writer")
    _autoloaded = False

    @classmethod
    def register_writer(cls, file_type: str, writer_cls: Type):
        cls._registry.add(file_type.lower(), writer_cls)

    @classmethod
    def _ensure_autoload(cls):
        if cls._autoloaded:
            return
        cls._autoload("src.core.knowledge_graph.writer")
        cls._autoloaded = True

    @classmethod
    def create_writer(cls, file_type: str, *args: Any, **kwargs):
        cls._ensure_autoload()
        writer_cls = cls._registry.get(file_type.lower())
        return writer_cls(*args, **kwargs)

    @classmethod
    def _autoload(cls, package_name: str):
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module_name)
