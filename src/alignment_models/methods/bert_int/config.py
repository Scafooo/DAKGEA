"""Configuration dataclasses for the BERT-INT alignment model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/models/bert_int.yaml"


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries, returning a fresh mapping."""
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(path: Optional[str]) -> Optional[Path]:
    """Return an absolute Path for the provided configuration entry."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


@dataclass
class PathsConfig:
    """Paths required by the BERT-INT pipeline."""

    cache_dir: Optional[str] = None
    dataset_root: Optional[str] = None
    description_dict: Optional[str] = None
    model_save_dir: Optional[str] = None
    model_save_prefix: Optional[str] = None

    def resolved(self) -> "PathsConfigResolved":
        """Return the same paths resolved to absolute `Path` instances."""
        return PathsConfigResolved(
            cache_dir=_resolve_path(self.cache_dir),
            dataset_root=_resolve_path(self.dataset_root),
            description_dict=_resolve_path(self.description_dict),
            model_save_dir=_resolve_path(self.model_save_dir),
            model_save_prefix=self.model_save_prefix,
        )


@dataclass
class PathsConfigResolved:
    """Resolved absolute paths for runtime usage."""

    cache_dir: Optional[Path]
    dataset_root: Optional[Path]
    description_dict: Optional[Path]
    model_save_dir: Optional[Path]
    model_save_prefix: Optional[str]


@dataclass
class DatasetConfig:
    """Dataset metadata retrieved from the configuration."""

    name: Optional[str] = None
    fold: Optional[str] = None


@dataclass
class BasicUnitConfig:
    """Hyper-parameters for training the basic BERT unit."""

    language: str = "ja"
    cuda_device: int = 0
    encoder_name: str = "bert-base-multilingual-cased"
    encoder_strategy: Optional[str] = None
    max_seq_length: int = 128
    dropout: float = 0.1
    model_input_dim: int = 768
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    epochs: int = 5
    batch_size: int = 24
    gradient_accumulation: int = 1
    load_pretrained: bool = True
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    random_divide_ill: bool = False
    train_ratio: Optional[float] = None
    seed: int = 11037
    negatives_per_positive: int = 2
    margin: float = 3.0
    eval_batch_size: int = 128
    eval_top_k: int = 1000
    candidate_top_k: int = 128
    candidate_batch_size: int = 128
    nearest_sample_num: int = 128
    result_size: int = 300
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BasicUnitConfig":
        dataset_cfg = DatasetConfig(**data.get("dataset", {}))
        kwargs = dict(data)
        kwargs["dataset"] = dataset_cfg
        return cls(**kwargs)


@dataclass
class InteractionConfig:
    """Hyper-parameters for the interaction model stage."""

    batch_size: int = 64
    cosine_top_k: int = 50
    temperature: float = 0.1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InteractionConfig":
        return cls(**data)


@dataclass
class SeedsConfig:
    """Seed values used across the BERT-INT pipeline."""

    global_: int = 42

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeedsConfig":
        return cls(global_=int(data.get("global", 42)))

    @property
    def global_seed(self) -> int:
        return self.global_


@dataclass
class BertIntConfig:
    """Root configuration payload for the BERT-INT alignment model."""

    device: str = "cpu"
    paths: PathsConfig = field(default_factory=PathsConfig)
    basic_unit: BasicUnitConfig = field(default_factory=BasicUnitConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    seed: SeedsConfig = field(default_factory=SeedsConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BertIntConfig":
        paths_cfg = PathsConfig(**data.get("paths", {}))
        basic_cfg = BasicUnitConfig.from_dict(data.get("basic_unit", {}))
        interaction_cfg = InteractionConfig.from_dict(data.get("interaction", {}))
        seed_cfg = SeedsConfig.from_dict(data.get("seed", {}))
        return cls(
            device=data.get("device", "cpu"),
            paths=paths_cfg,
            basic_unit=basic_cfg,
            interaction=interaction_cfg,
            seed=seed_cfg,
        )

    @property
    def resolved_paths(self) -> PathsConfigResolved:
        """Return the resolved, absolute paths."""
        return self.paths.resolved()

    def validate(self) -> None:
        """Perform lightweight validation of the configuration values."""
        encoder_name = self.basic_unit.encoder_name
        if not encoder_name:
            raise ValueError("basic_unit.encoder_name must not be empty.")
        if self.paths.dataset_root is None:
            logger.warning("No dataset_root configured for BERT-INT basic unit.")


def load_bert_int_config(
    *,
    path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> BertIntConfig:
    """Load the BERT-INT configuration from YAML plus optional overrides."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"BERT-INT configuration file not found: {config_path}")

    payload = load_yaml(config_path)
    model_section = payload.get("model", {})
    if overrides:
        model_section = _merge_dicts(model_section, dict(overrides))

    config = BertIntConfig.from_dict(model_section)
    config.validate()
    return config


__all__ = (
    "BertIntConfig",
    "BasicUnitConfig",
    "InteractionConfig",
    "PathsConfig",
    "PathsConfigResolved",
    "DatasetConfig",
    "SeedsConfig",
    "load_bert_int_config",
)
