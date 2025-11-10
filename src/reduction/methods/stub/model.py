"""No-op reduction strategy for testing purposes."""

from __future__ import annotations

from src.core.dataset import Dataset
from src.logger import get_logger
from src.reduction.registry import REDUCTION_REGISTRY

logger = get_logger(__name__)


@REDUCTION_REGISTRY.register("stub")
class StubReducer:
    """Return the incoming dataset unchanged."""

    def __init__(self, config):
        self.config = config or {}

    def reduce(self, dataset: Dataset) -> Dataset:
        logger.debug("[StubReducer] Returning dataset unchanged.")
        return dataset


__all__ = ["StubReducer"]
