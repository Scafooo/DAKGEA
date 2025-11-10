"""Helper utilities for discovering registered reducers and augmenters."""

from src.augmentation.registry import load_builtin_augmentations
from src.config.loader import Config
from src.logger import get_logger
from src.reduction.registry import load_builtin_reducers

logger = get_logger(__name__)


def autoload_registries() -> None:
    """Ensure reduction and augmentation plugins are discoverable."""

    cfg = Config().get().get("experiment_defaults", {})
    disabled = set(cfg.get("disabled_registries", []))

    if "reducers" not in disabled:
        load_builtin_reducers()
    else:
        logger.debug("Skipping reducer autoload (disabled by config).")

    if "augmenters" not in disabled:
        load_builtin_augmentations()
    else:
        logger.debug("Skipping augmenter autoload (disabled by config).")
