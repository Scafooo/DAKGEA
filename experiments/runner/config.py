"""Experiment configuration loaders."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_experiment_cfg(config_path: Path) -> Dict[str, Any]:
    """Read the experiment YAML file and return the nested configuration dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if "experiment" not in payload:
        raise KeyError("Top-level key 'experiment' missing from configuration file.")
    return payload["experiment"]
