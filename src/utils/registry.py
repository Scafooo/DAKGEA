"""Reusable registry utilities with optional auto-discovery support."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Register callable/classes under string keys with optional module auto-loading."""

    def __init__(self, item_kind: str) -> None:
        self._item_kind = item_kind
        self._registry: Dict[str, T] = {}

    def register(self, name: str):
        """Decorator that registers the given class or factory under ``name``."""

        def decorator(item: T) -> T:
            self.add(name, item)
            return item

        return decorator

    def add(self, name: str, item: T) -> None:
        """Register ``cls`` under ``name`` immediately."""
        key = name
        if key in self._registry:
            raise ValueError(f"{self._item_kind} '{name}' already registered.")
        self._registry[key] = item

    def get(self, name: str) -> T:
        """Return the registered item, raising if missing."""
        if name not in self._registry:
            raise ValueError(f"{self._item_kind} '{name}' not registered.")
        return self._registry[name]

    def list(self) -> Iterable[str]:
        """Return all registered names."""
        return list(self._registry.keys())

    def items(self) -> Iterable[tuple[str, T]]:
        """Return (name, item) pairs."""
        return self._registry.items()

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
