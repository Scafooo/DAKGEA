"""Factory utilities for knowledge-graph readers."""

import importlib
from typing import Any, Type

from src.util.registry import Registry


class KnowledgeGraphReaderFactory:
    _registry = Registry("Knowledge graph reader")

    @classmethod
    def register_reader(cls, file_type: str, reader_cls: Type) -> None:
        cls._registry.add(file_type.lower(), reader_cls)

    @classmethod
    def create_reader(cls, file_type: str, *args: Any, **kwargs):
        key = file_type.lower()
        try:
            reader_cls = cls._registry.get(key)
        except ValueError:
            _ensure_registered(key)
            reader_cls = cls._registry.get(key)
        return reader_cls(*args, **kwargs)


_READER_MODULES = {
    "bert_int": "src.core.knowledge_graph.reader.bert_int_knowledge_graph_reader",
    "openea": "src.core.knowledge_graph.reader.openea_knowledge_graph_reader",
    "rdf": "src.core.knowledge_graph.reader.rdf_knowledge_graph_reader",
}


def _ensure_registered(file_type: str) -> None:
    module_path = _READER_MODULES.get(file_type)
    if module_path:
        importlib.import_module(module_path)
