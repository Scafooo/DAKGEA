"""Registry utilities for pluggable alignment models."""

import importlib
from typing import Iterable

from src.util.registry import Registry

MODEL_REGISTRY: Registry[type] = Registry("Alignment model")

_BUILTIN_MODEL_MODULES: Iterable[str] = (
    "src.alignment_models.methods.stub",      # smoke-test model
    "src.alignment_models.methods.bert_int",  # BERT-INT integration
    "src.alignment_models.methods.hybea",     # HybEA integration
)


def load_builtin_models() -> None:
    """Import built-in alignment model modules so they register themselves."""
    for module_path in _BUILTIN_MODEL_MODULES:
        importlib.import_module(module_path)
