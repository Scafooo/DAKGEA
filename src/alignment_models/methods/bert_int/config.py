"""Configuration helpers for the BERT-INT alignment model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BasicUnitConfig:
    """Hyper-parameters for the basic BERT unit encoder."""

    encoder_name: str = "bert-base-multilingual-cased"
    encoder_strategy: str = "auto"
    input_dim: int = 768
    output_dim: int = 300
    epochs: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 24
    test_batch_size: int = 128
    max_seq_length: int = 128
    negatives: int = 2
    margin: float = 3.0
    random_divide_ill: bool = False
    train_ill_rate: float = 0.3
    candidate_topk: int = 128


@dataclass
class InteractionConfig:
    """Hyper-parameters for the interaction-based scoring network."""

    epochs: int = 200
    learning_rate: float = 5e-4
    batch_size: int = 128
    negatives: int = 5
    margin: float = 1.0
    candidate_topk: int = 1000
    kernel_num: int = 21
    neighbor_max: int = 50
    attribute_max: int = 50
    hidden_dim: int = 11


@dataclass
class PathsConfig:
    """Filesystem locations required by BERT-INT."""

    work_dir: str = "experiments/bert_int"
    models_dir: Optional[str] = None
    cache_dir: Optional[str] = None


@dataclass
class BertIntConfig:
    """Full configuration payload for the BERT-INT alignment model."""

    seed: int = 11037
    device: str = "cuda:0"
    language: str = "en"
    candidate_num: int = 1000
    basic_unit: BasicUnitConfig = field(default_factory=BasicUnitConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BertIntConfig":
        """Build a config instance from a nested dictionary."""

        basic_data = data.get("basic_unit", {})
        interaction_data = data.get("interaction", {})
        paths_data = data.get("paths", {})

        cfg = cls(
            seed=data.get("seed", cls.seed),
            device=data.get("device", cls.device),
            language=data.get("language", cls.language),
            candidate_num=data.get("candidate_num", cls.candidate_num),
            basic_unit=BasicUnitConfig(**basic_data),
            interaction=InteractionConfig(**interaction_data),
            paths=PathsConfig(**paths_data),
        )
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain dictionary."""

        return {
            "seed": self.seed,
            "device": self.device,
            "language": self.language,
            "candidate_num": self.candidate_num,
            "basic_unit": vars(self.basic_unit),
            "interaction": vars(self.interaction),
            "paths": vars(self.paths),
        }
