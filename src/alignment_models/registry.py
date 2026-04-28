"""Registry utilities for pluggable alignment models."""

import importlib
from typing import Dict

from src.utils.registry import Registry

MODEL_REGISTRY: Registry[type] = Registry("Alignment model")

# Map registry keys to the module that registers them
_MODEL_MODULES: Dict[str, str] = {
    "stub": "src.alignment_models.methods.stub",
    "hybea": "src.alignment_models.methods.hybea.model",
    "bert_int": "src.alignment_models.methods.bert_int.model",
    "rrea": "src.alignment_models.methods.RREA.model",
    "multiKE": "src.alignment_models.methods.multiKE.model",
    "attrE": "src.alignment_models.methods.attrE.model",
    "sdea": "src.alignment_models.methods.sdea.model",
}


def _ensure_registered(model_name: str) -> None:
    module_path = _MODEL_MODULES.get(model_name)
    if module_path:
        importlib.import_module(module_path)


def get_alignment_model(model_name: str):
    """Return the registered model class, importing lazily when necessary."""
    try:
        return MODEL_REGISTRY.get(model_name)
    except ValueError as exc:
        _ensure_registered(model_name)
        return MODEL_REGISTRY.get(model_name)


def load_builtin_models() -> None:
    """Import bundled alignment models except optional heavy integrations."""
    for name in ("stub", "hybea", "bert_int", "rrea", "multiKE", "attrE"):
        _ensure_registered(name)
