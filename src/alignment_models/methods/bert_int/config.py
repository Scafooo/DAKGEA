"""Configuration utilities for the BERT-INT alignment model."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/models/bert_int.yaml"

PATH_KEYS = {"cache_dir", "dataset_root", "description_dict", "model_save_dir"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "device": "cpu",
    "paths": {
        "cache_dir": "results/cache/bert_int",
        "dataset_root": None,
        "description_dict": "Bert-int-pure/2016-10-des_dict.pkl",
        "model_save_dir": "results/bert_int/basic_unit",
        "model_save_prefix": "run",
    },
    "basic_unit": {
        "encoder_name": "bert-base-multilingual-cased",
        "encoder_strategy": "auto",
        "max_seq_length": 128,
        "dropout": 0.1,
        "model_input_dim": 768,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.0,
        "epochs": 5,
        "batch_size": 24,
        "gradient_accumulation": 1,
        "load_pretrained": True,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "random_divide_ill": False,
        "train_ratio": None,
        "seed": 11037,
        "negatives_per_positive": 2,
        "margin": 3.0,
        "eval_batch_size": 128,
        "eval_top_k": 1000,
        "candidate_top_k": 128,
        "candidate_batch_size": 128,
        "nearest_sample_num": 128,
        "result_size": 300,
        "cuda_device": 0,
        "language": None,
        "dataset": {
            "name": None,
            "fold": None,
        },
    },
    "interaction": {
        "batch_size": 64,
        "cosine_top_k": 50,
        "temperature": 0.1,
    },
    "seed": {
        "global": 42,
    },
}


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries, returning a new mapping."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_paths(paths: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    for key, value in paths.items():
        if key not in PATH_KEYS or value in (None, ""):
            resolved[key] = value
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        resolved[key] = str(candidate)
    return resolved


def normalise_bert_int_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge raw configuration data with defaults and resolve paths."""
    config = _deep_merge(DEFAULT_CONFIG, raw or {})
    config["paths_resolved"] = _resolve_paths(config.get("paths", {}))
    return config


def load_bert_int_config(
    *,
    path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load the BERT-INT configuration from YAML plus optional overrides."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"BERT-INT configuration file not found: {config_path}")

    payload = load_yaml(config_path) or {}
    model_section = payload.get("model", {})
    if overrides:
        model_section = _deep_merge(model_section, overrides)

    config = normalise_bert_int_config(model_section)
    if not config["basic_unit"].get("encoder_name"):
        raise ValueError("basic_unit.encoder_name must not be empty.")
    if not config["paths"].get("dataset_root"):
        logger.debug("No dataset_root configured for BERT-INT basic unit.")
    return config


__all__ = (
    "DEFAULT_CONFIG",
    "load_bert_int_config",
    "normalise_bert_int_config",
)
