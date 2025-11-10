"""Base classes and helpers for dataset augmentation methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.config.loader import PROJECT_ROOT
from src.logger import get_logger, get_structured_logger


class AugmentationMethod:
    """Shared helpers for augmentation strategies."""

    registry_name: Optional[str] = None

    def __init__(self, config: Optional[dict] = None, *, name: Optional[str] = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.name = name or self.registry_name or self.__class__.__name__.lower()
        self.logger = get_logger(f"augmentation.{self.name}")
        self.slogger = get_structured_logger(f"augmentation.{self.name}")
        self.external_root = (PROJECT_ROOT / "data" / "external").resolve()
        self.external_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def resolve_external_dir(self, *parts: str, create: bool = True) -> Path:
        """Return a directory under data/external, creating it when requested."""
        path = self.external_root.joinpath(*parts)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def section(self, title: str) -> None:
        """Log a major step in the augmentation flow."""
        self.slogger.section(title)

    def subsection(self, title: str) -> None:
        """Log a sub-step during augmentation."""
        self.slogger.subsection(title)

    # ------------------------------------------------------------------
    # To be implemented by subclasses
    # ------------------------------------------------------------------
    def augment(self, dataset):  # pragma: no cover - abstract method
        raise NotImplementedError
