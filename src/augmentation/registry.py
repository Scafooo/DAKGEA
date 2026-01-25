"""Registry utilities for augmentation strategies."""

import importlib
from typing import Iterable

from src.utils.registry import Registry

AUGMENTATION_REGISTRY: Registry[type] = Registry("Augmentation")

_BUILTIN_AUGMENTATION_MODULES: Iterable[str] = (
    "src.augmentation.methods.plm.augmenter",
    "src.augmentation.methods.plm_mixup.augmenter",
    "src.augmentation.methods.stub.augmenter",
)


def load_builtin_augmentations() -> None:
    """Import built-in augmentation modules so they self-register."""
    for module_path in _BUILTIN_AUGMENTATION_MODULES:
        importlib.import_module(module_path)
