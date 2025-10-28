from typing import Type, Dict
import importlib
import pkgutil


class AugmentationRegistry:
    """Registry for Data Augmentation methods applied to Knowledge Graph datasets."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register an augmentation method with a given name."""
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"Augmentation '{name}' already registered.")
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        """Retrieve an augmentation class by name."""
        if name not in self._registry:
            raise ValueError(f"Augmentation '{name}' not registered.")
        return self._registry[name]

    def list(self):
        """List all registered augmentations."""
        return list(self._registry.keys())

    def autoload(self, package: str = "src.augmentation.methods"):
        """Automatically import all augmentation methods under the given package."""
        pkg = importlib.import_module(package)
        for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{package}.{modname}")


# Global singleton
AUGMENTATION_REGISTRY = AugmentationRegistry()
