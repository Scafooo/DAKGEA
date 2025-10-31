"""Augmentation package public API."""

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY

__all__ = ["AugmentationMethod", "AUGMENTATION_REGISTRY"]
