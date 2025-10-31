"""Registry utilities for dataset reduction strategies."""

from src.util.registry import Registry


class ReductionRegistry(Registry):
    """Registry for dataset reduction methods."""

    def __init__(self) -> None:
        super().__init__("Reduction method")

    def autoload(self, package_name: str = "src.reduction.methods") -> None:
        """Import all submodules in a package so that decorated classes register themselves."""
        super().autoload(package_name, recursive=True)


REDUCTION_REGISTRY = ReductionRegistry()
