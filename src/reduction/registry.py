import importlib
import pkgutil
from typing import Type, Dict

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

    # 🔹 NEW: autoload all submodules under a given package
    def autoload(self, package_name: str):
        """Dynamically import all submodules in a package to populate registry."""
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return  # Not a package

        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module_name)

# Global instance
REDUCTION_REGISTRY = ReductionRegistry()
