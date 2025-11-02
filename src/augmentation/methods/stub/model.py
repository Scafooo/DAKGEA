"""No-op augmentation strategy, useful for smoke tests."""

from __future__ import annotations

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.logger import get_logger

logger = get_logger(__name__)


@AUGMENTATION_REGISTRY.register("stub")
class StubAugmentation(AugmentationMethod):
    """Return the reduced dataset without modifications."""

    def augment(self, dataset):
        logger.debug("[StubAugmentation] Returning dataset unchanged.")
        return dataset


__all__ = ("StubAugmentation",)
