"""Helper utilities for discovering registered reducers, augmenters, and models."""

from src.alignment_models.registry import MODEL_REGISTRY, load_builtin_models
from src.augmentation.registry import AUGMENTATION_REGISTRY, load_builtin_augmentations
from src.reduction.registry import REDUCTION_REGISTRY, load_builtin_reducers


def autoload_registries() -> None:
    """Ensure all reduction, augmentation, and model plugins are discoverable."""
    load_builtin_reducers()
    load_builtin_augmentations()
    load_builtin_models()
