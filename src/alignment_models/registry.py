"""Registry utilities for pluggable alignment models."""

import importlib
import pkgutil
from typing import Dict, Type

class ModelRegistry:
    """Registry for alignment models (e.g., HybEA, Knowformer, BERT-INT, etc.)."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register an alignment model with a given name."""
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"Alignment model '{name}' already registered.")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        """Retrieve an alignment model by name."""
        if name not in self._registry:
            raise ValueError(f"Alignment model '{name}' not registered.")
        return self._registry[name]

    def list(self):
        """List all registered models."""
        return list(self._registry.keys())

    def autoload(self, package: str = "src.alignment_models.methods"):
        """Automatically import all model modules under the given package."""
        pkg = importlib.import_module(package)
        for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{package}.{modname}")

MODEL_REGISTRY = ModelRegistry()
