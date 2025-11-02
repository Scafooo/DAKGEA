"""Factory utilities for knowledge-graph writers."""

import importlib
from typing import Any, Type

from src.util.registry import Registry


class KnowledgeGraphWriterFactory:
    _registry = Registry("Knowledge graph writer")

    @classmethod
    def register_writer(cls, file_type: str, writer_cls: Type) -> None:
        cls._registry.add(file_type.lower(), writer_cls)

    @classmethod
    def create_writer(cls, file_type: str, *args: Any, **kwargs):
        key = file_type.lower()
        try:
            writer_cls = cls._registry.get(key)
        except ValueError:
            _ensure_registered(key)
            writer_cls = cls._registry.get(key)
        return writer_cls(*args, **kwargs)


_WRITER_MODULES = {
    "hybea": "src.core.knowledge_graph.writer.hybea_knowledge_graph_writer",
    "rdf": "src.core.knowledge_graph.writer.rdf_knowledge_graph_writer",
}


def _ensure_registered(file_type: str) -> None:
    module_path = _WRITER_MODULES.get(file_type)
    if module_path:
        importlib.import_module(module_path)
