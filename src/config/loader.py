import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_dicts(d1: dict, d2: dict) -> dict:
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1

class Config:
    def __init__(self, global_path=None, model_path=None, exp_path=None):
        self.cfg = {}

        # Load global config
        if global_path is None:
            global_path = PROJECT_ROOT / "config/global.yaml"
        self.cfg = load_yaml(global_path)

        # Merge model config
        if model_path:
            model_cfg = load_yaml(model_path)
            merge_dicts(self.cfg, model_cfg)

        # Merge experiment config
        if exp_path:
            exp_cfg = load_yaml(exp_path)
            merge_dicts(self.cfg, exp_cfg)

        self.resolve_paths()

    def resolve_paths(self):
        paths = self.cfg.get("paths", {})
        for key, val in paths.items():
            paths[key] = (PROJECT_ROOT / val).resolve()

    def get(self):
        return self.cfg
