"""Configuration utilities for the RREA alignment model."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/models/rrea.yaml"

PATH_KEYS = {"cache_dir", "model_save_dir"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "device": "cpu",
    "paths": {
        "cache_dir": "results/cache/rrea",
        "model_save_dir": "results/rrea/checkpoints",
        "model_save_prefix": "rrea",
    },
    "model": {
        # Graph Attention parameters
        "embedding_dim": 100,       # Dimension of entity embeddings
        "depth": 2,                 # Number of GAT layers
        "attn_heads": 1,            # Number of attention heads
        "attn_heads_reduction": "concat",  # 'concat' or 'average'
        "dropout_rate": 0.3,        # Dropout rate
        "activation": "relu",       # Activation function
        "use_bias": False,          # Use bias in layers
        "use_w": False,             # Use weight matrix

        # Training parameters
        "epochs": 1200,             # Number of training epochs
        "batch_size": 2500,         # Training batch size
        "learning_rate": 0.001,     # Learning rate
        "optimizer": "adam",        # Optimizer type
        "gamma": 3.0,               # Margin for loss function
        "neg_num": 5,               # Number of negative samples
        "train_ratio": 0.3,         # Ratio of training samples

        # CSLS parameters
        "csls_k": 10,               # K for CSLS scoring
        "use_csls": True,           # Use CSLS or cosine similarity
        "num_threads": 16,          # Number of threads for evaluation

        # Evaluation parameters
        "eval_frequency": 100,      # Evaluate every N epochs
        "eval_top_k": [1, 5, 10, 50],  # Top-K for evaluation
        "early_stop": True,         # Enable early stopping
        "early_stop_patience": 50,  # Patience for early stopping

        # Regularization
        "weight_decay": 0.0,        # L2 regularization
        "max_grad_norm": 5.0,       # Gradient clipping

        # Advanced options
        "normalize_embeddings": True,  # L2 normalize embeddings
        "relational_reflection": True, # Use relational reflection
    },
}


def normalize_rrea_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize RREA configuration, resolving paths and setting defaults.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized configuration with resolved paths
    """
    config = copy.deepcopy(config)

    # Resolve paths relative to PROJECT_ROOT
    if "paths" in config:
        for key in PATH_KEYS:
            if key in config["paths"] and config["paths"][key]:
                path_val = config["paths"][key]
                if not Path(path_val).is_absolute():
                    config["paths"][key] = str(PROJECT_ROOT / path_val)

    # Set device from top-level if specified
    if "device" in config and "model" in config:
        if "device" not in config["model"]:
            config["model"]["device"] = config["device"]

    return config


def load_rrea_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and merge RREA configuration from YAML and overrides.

    Args:
        config_path: Path to YAML configuration file (default: config/models/rrea.yaml)
        overrides: Dictionary of configuration overrides

    Returns:
        Merged and normalized configuration
    """
    # Start with default config
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Load from YAML if exists
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if config_path and config_path.exists():
        yaml_config = load_yaml(config_path)
        if yaml_config:
            _deep_update(config, yaml_config)
            logger.info(f"[RREA] Loaded configuration from {config_path}")
    else:
        logger.info("[RREA] Using default configuration (no YAML file found)")

    # Apply overrides
    if overrides:
        _deep_update(config, overrides)
        logger.info("[RREA] Applied configuration overrides")

    # Normalize paths and resolve device
    config = normalize_rrea_config(config)

    return config


def _deep_update(base: Dict[str, Any], update: Mapping[str, Any]) -> None:
    """Recursively update nested dictionary.

    Args:
        base: Base dictionary to update (modified in-place)
        update: Dictionary with updates
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
