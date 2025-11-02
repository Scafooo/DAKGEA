"""Configuration helpers for the BERT-INT integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

logger = get_logger(__name__)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``override`` into ``base`` returning a fresh dictionary."""
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def resolve_device(preferred: Optional[str]) -> torch.device:
    """Return the most appropriate torch device based on configuration."""
    if preferred:
        desired = preferred.lower()
        if desired.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(preferred)
            logger.warning(
                "[BERT-INT] Requested CUDA device '%s' is unavailable, falling back to CPU.",
                preferred,
            )
        elif desired == "cpu":
            return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BertIntPaths:
    cache_dir: Path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BertIntPaths":
        cache_dir = payload.get("cache_dir", "results/cache/bert_int")
        cache_path = (PROJECT_ROOT / cache_dir).resolve()
        return cls(cache_dir=cache_path)


@dataclass
class BertIntBasicUnitSettings:
    encoder_name: str = "bert-base-multilingual-cased"
    max_seq_length: int = 128
    dropout: float = 0.1
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 16
    gradient_accumulation: int = 1
    load_pretrained: bool = True
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    train_ratio: float = 0.8
    negatives_per_positive: int = 1
    margin: float = 1.0
    eval_batch_size: int = 128

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BertIntBasicUnitSettings":
        data = {**cls().__dict__, **(payload or {})}
        return cls(
            encoder_name=data["encoder_name"],
            max_seq_length=int(data["max_seq_length"]),
            dropout=float(data["dropout"]),
            learning_rate=float(data["learning_rate"]),
            weight_decay=float(data["weight_decay"]),
            epochs=int(data["epochs"]),
            batch_size=int(data["batch_size"]),
            gradient_accumulation=max(1, int(data["gradient_accumulation"])),
            load_pretrained=bool(data["load_pretrained"]),
            warmup_steps=int(data["warmup_steps"]),
            max_grad_norm=float(data["max_grad_norm"]),
            train_ratio=float(data["train_ratio"]),
            negatives_per_positive=max(1, int(data["negatives_per_positive"])),
            margin=float(data["margin"]),
            eval_batch_size=int(data["eval_batch_size"]),
        )


@dataclass
class BertIntInteractionSettings:
    batch_size: int = 64
    cosine_top_k: int = 50
    temperature: float = 0.1

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BertIntInteractionSettings":
        data = {**cls().__dict__, **(payload or {})}
        return cls(
            batch_size=int(data["batch_size"]),
            cosine_top_k=int(data["cosine_top_k"]),
            temperature=float(data["temperature"]),
        )


@dataclass
class BertIntConfig:
    device: torch.device
    paths: BertIntPaths
    basic_unit: BertIntBasicUnitSettings
    interaction: BertIntInteractionSettings
    seed: int

    @classmethod
    def from_model_section(
        cls,
        model_section: Dict[str, Any],
    ) -> "BertIntConfig":
        device = resolve_device(model_section.get("device"))
        paths = BertIntPaths.from_dict(model_section.get("paths", {}))
        basic_unit = BertIntBasicUnitSettings.from_dict(model_section.get("basic_unit", {}))
        interaction = BertIntInteractionSettings.from_dict(model_section.get("interaction", {}))
        seed_cfg = model_section.get("seed", {})
        seed = int(seed_cfg.get("global", 42))
        return cls(
            device=device,
            paths=paths,
            basic_unit=basic_unit,
            interaction=interaction,
            seed=seed,
        )


def load_bert_int_config(stage_override: Optional[Dict[str, Any]] = None) -> BertIntConfig:
    """Load the BERT-INT configuration from YAML files plus stage overrides."""
    base_path = PROJECT_ROOT / "config/models/bert_int.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"BERT-INT model configuration not found: {base_path}")
    model_section = load_yaml(base_path).get("model", {})

    local_override = PROJECT_ROOT / "config/models/bert_int.local.yaml"
    if local_override.exists():
        model_section = _merge_dicts(model_section, load_yaml(local_override).get("model", {}))

    stage_override_path = PROJECT_ROOT / "config/models/bert_int.stage.yaml"
    if stage_override_path.exists():
        model_section = _merge_dicts(model_section, load_yaml(stage_override_path).get("model", {}))

    if stage_override:
        model_section = _merge_dicts(model_section, stage_override)

    return BertIntConfig.from_model_section(model_section)


__all__ = [
    "BertIntBasicUnitSettings",
    "BertIntConfig",
    "BertIntInteractionSettings",
    "BertIntPaths",
    "load_bert_int_config",
    "resolve_device",
]
