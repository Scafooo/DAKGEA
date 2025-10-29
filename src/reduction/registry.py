"""Registry utilities for dataset reduction strategies."""

import importlib
import pkgutil
from typing import Dict, Type

class ReductionRegistry:
    """Registry for dataset reduction methods."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register a reduction class with a given name."""
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"Reduction method {name} already registered.")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        """Retrieve a reduction class by name."""
        if name not in self._registry:
            raise ValueError(f"Reduction method {name} is not registered.")
        return self._registry[name]

    def autoload(self, package_name: str = "src.reduction.methods") -> None:
        """Import all submodules in a package so that decorated classes register themselves."""
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return

        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, f"{package.__name__}."
        ):
            importlib.import_module(module_name)

REDUCTION_REGISTRY = ReductionRegistry()
