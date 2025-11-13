"""Reproducibility utilities for ensuring deterministic behavior across runs."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def _load_global_config() -> dict:
    """Load global configuration to check deterministic flag.

    Returns:
        Global configuration dictionary
    """
    try:
        # Find project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent
        global_config_path = project_root / "config" / "global.yaml"

        if global_config_path.exists():
            with open(global_config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    except Exception as e:
        logger.warning(f"Failed to load global config for reproducibility settings: {e}")
        return {}


def is_deterministic_mode() -> bool:
    """Check if deterministic mode is enabled in global configuration.

    Returns:
        True if deterministic mode is enabled, False otherwise
    """
    global_cfg = _load_global_config()
    return global_cfg.get("reproducibility", {}).get("deterministic", True)


def set_random_seeds(
    seed: int,
    deterministic: Optional[bool] = None
) -> None:
    """Set random seeds for all libraries to ensure reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU and CUDA)
    - Python hash seed (PYTHONHASHSEED) for deterministic hash-based operations

    Additionally, if deterministic mode is enabled, it configures PyTorch's
    cuDNN backend for deterministic operations, which may reduce performance
    by ~10-20% but ensures identical results across runs.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode. If None, reads
                      from global configuration (default: None)

    Examples:
        >>> set_random_seeds(42)  # Uses global config for deterministic setting
        >>> set_random_seeds(42, deterministic=True)  # Force deterministic mode
        >>> set_random_seeds(42, deterministic=False)  # Force non-deterministic mode
    """
    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine if we should enable deterministic mode
    if deterministic is None:
        deterministic = is_deterministic_mode()

    # Set PYTHONHASHSEED for deterministic hash-based operations
    # This affects dict/set iteration order and is critical for RDFlib graph iteration
    if deterministic:
        # Check if PYTHONHASHSEED was already set before Python started
        # If not set, warn that it should be set before process start
        current_hashseed = os.environ.get("PYTHONHASHSEED")
        if current_hashseed is None or current_hashseed == "random":
            logger.warning(
                "[Reproducibility] PYTHONHASHSEED not set. For full reproducibility, "
                "set PYTHONHASHSEED=%d before starting Python. "
                "Setting it now (may not affect all operations).",
                seed
            )
        # Set it anyway for any child processes
        os.environ["PYTHONHASHSEED"] = str(seed)

    # Configure cuDNN for deterministic operations if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(
            "[Reproducibility] Deterministic mode enabled (seed=%d). "
            "Note: This may reduce performance by ~10-20%%.",
            seed
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.debug(
            "[Reproducibility] Non-deterministic mode (seed=%d). "
            "Results may vary slightly across runs for better performance.",
            seed
        )
