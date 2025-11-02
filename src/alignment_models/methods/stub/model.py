"""Minimal alignment model used for smoke tests."""

from __future__ import annotations

from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__)


@MODEL_REGISTRY.register("stub")
class StubAlignment:
    """Return constant metrics without performing any computation."""

    MODEL_CONFIG_PATH = PROJECT_ROOT / "config/models/stub.yaml"

    def __init__(self, config):
        self.stage_config = config or {}

    def evaluate(self, dataset_reduced, dataset_augmented):
        logger.info("[STUB] Evaluating dataset (no-op)")
        return {"hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}


__all__ = ("StubAlignment",)
