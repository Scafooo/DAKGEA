"""Registry utilities for dataset reduction strategies."""

import importlib
from typing import Iterable

from src.util.registry import Registry

REDUCTION_REGISTRY: Registry[type] = Registry("Reduction method")

_BUILTIN_REDUCTION_MODULES: Iterable[str] = (
    "src.reduction.methods.random_entities",  # module registering RandomEntitiesReducer
    "src.reduction.methods.stub",              # stub reducer
)


def load_builtin_reducers() -> None:
    """Import the built-in reduction modules so they register with the registry."""
    for module_path in _BUILTIN_REDUCTION_MODULES:
        importlib.import_module(module_path)
