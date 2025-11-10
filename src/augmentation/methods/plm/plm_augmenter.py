"""PLM-based augmentation strategy leveraging latent interpolation."""

from __future__ import annotations

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via BART-based latent interpolation of literal attributes."""

    registry_name = "plm_augmentation"

    def __init__(self, config):
        super().__init__(config)

    def augment(self, dataset: Dataset) -> Dataset:
        pass
