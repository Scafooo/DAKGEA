"""Reusable registry utilities with optional auto-discovery support."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type


class Registry:
    """Register callable/classes under string keys with optional module auto-loading."""

    def __init__(self, item_kind: str) -> None:
        self._item_kind = item_kind
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator that registers the given class or factory under ``name``."""

        def decorator(cls: Type):
            self.add(name, cls)
            return cls

        return decorator

    def add(self, name: str, cls: Type) -> None:
        """Register ``cls`` under ``name`` immediately."""
        key = name
        if key in self._registry:
            raise ValueError(f"{self._item_kind} '{name}' already registered.")
        self._registry[key] = cls

    def get(self, name: str) -> Type:
        """Return the registered item, raising if missing."""
        if name not in self._registry:
            raise ValueError(f"{self._item_kind} '{name}' not registered.")
        return self._registry[name]

    def list(self):
        """Return all registered names."""
        return list(self._registry.keys())

    def autoload(self, package: str, *, recursive: bool = False) -> None:
        """Import all modules in ``package`` so decorated classes self-register."""
        module = importlib.import_module(package)
        if not hasattr(module, "__path__"):
            return

        if recursive:
            iterator = pkgutil.walk_packages(module.__path__, f"{module.__name__}.")
            for _, module_name, _ in iterator:
                importlib.import_module(module_name)
        else:
            for _, modname, _ in pkgutil.iter_modules(module.__path__):
                importlib.import_module(f"{package}.{modname}")
