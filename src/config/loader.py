"""Utility helpers for loading and merging configuration files."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries, giving precedence to values in `extra`."""
    for key, value in extra.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

class Config:
    """Loader that merges global, model, and experiment YAML configuration files."""

    def __init__(
        self,
        global_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        exp_path: Optional[Path] = None,
    ):
        self.cfg: Dict[str, Any] = {}

        if global_path is None:
            global_path = PROJECT_ROOT / "config/global.yaml"
        self.cfg = load_yaml(global_path)

        if model_path:
            model_cfg = load_yaml(model_path)
            self.cfg = merge_dicts(self.cfg, model_cfg)

        if exp_path:
            exp_cfg = load_yaml(exp_path)
            self.cfg = merge_dicts(self.cfg, exp_cfg)

        self.resolve_paths()

    def resolve_paths(self):
        """Convert relative paths in the `paths` section to absolute filesystem paths."""
        paths = self.cfg.get("paths", {})
        for key, val in paths.items():
            paths[key] = (PROJECT_ROOT / val).resolve()

    def get(self):
        """Return the merged configuration dictionary."""
        return self.cfg
