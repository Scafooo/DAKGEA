"""Registry utilities for augmentation strategies."""

from src.util.registry import Registry


class AugmentationRegistry(Registry):
    """Registry for Data Augmentation methods applied to Knowledge Graph datasets."""

    def __init__(self) -> None:
        super().__init__("Augmentation")

    def autoload(self, package: str = "src.augmentation.methods") -> None:
        """Automatically import all augmentation methods under the given package."""
        super().autoload(package, recursive=False)


AUGMENTATION_REGISTRY = AugmentationRegistry()
