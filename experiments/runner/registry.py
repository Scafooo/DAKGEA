"""Helper utilities for discovering registered reducers, augmenters, and models."""

from src.alignment_models.registry import MODEL_REGISTRY
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.reduction.registry import REDUCTION_REGISTRY


def autoload_registries() -> None:
    """Ensure all reduction, augmentation, and model plugins are discoverable."""
    REDUCTION_REGISTRY.autoload()
    AUGMENTATION_REGISTRY.autoload()
    MODEL_REGISTRY.autoload()
