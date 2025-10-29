"""Structured configuration helpers for the HybEA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class AttributeConfig:
    """Settings for the attribute modelling stage."""

    model_input_dim: int = 768
    epochs: int = 200
    learning_rate: float = 1e-5
    train_batch_size: int = 24
    test_batch_size: int = 128
    negatives: int = 2
    margin: float = 3.0
    nearest_sample_num: int = 128
    candidate_generator_batch_size: int = 128
    csls: int = 2


@dataclass
class StructureConfig:
    """Settings for the structural modelling stage (KnowFormer / RREA)."""

    enabled: bool = True
    random_initialization: bool = False
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    input_dropout_prob: float = 0.5
    attention_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.3
    residual_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    residual_w: float = 0.5
    addition_loss_w: float = 0.1
    relation_combine_dropout_prob: float = 0.2
    epochs: int = 200
    min_epochs: int = 10
    learning_rate: float = 5e-4
    batch_size: int = 2048
    eval_batch_size: int = 4096
    early_stop_max_times: int = 3
    soft_label: float = 0.25
    eval_freq: int = 5
    start_eval: int = 0
    swa_pre_num: int = 5
    use_gelu: bool = False


@dataclass
class DatasetSpec:
    """Dataset-specific hyper-parameters required by the legacy pipeline."""

    candidate_topk: int
    source_input_size: int
    target_input_size: int


@dataclass
class HybeaConfig:
    """Full configuration payload used to drive the HybEA pipeline."""

    device: str = "cuda"
    mode: str = "Hybea"
    structural_model: str = "Knowformer"
    reduction_ratio: float = 1.0
    pipeline_seed: int = 42
    attribute_seed: int = 11037
    attribute: AttributeConfig = field(default_factory=AttributeConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    datasets: Dict[str, DatasetSpec] = field(default_factory=dict)
    results_dir: str = "experiments/hybea"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HybeaConfig":
        """Create a configuration instance from a raw dictionary."""

        seeds = data.get("seeds", {})
        attribute_cfg = AttributeConfig(**data.get("attribute", {}))
        structure_cfg = StructureConfig(**data.get("structure", {}))

        dataset_specs = {}
        for name, spec in data.get("datasets", {}).items():
            dataset_specs[name] = DatasetSpec(**spec)

        return cls(
            device=data.get("device", cls.device),
            mode=data.get("mode", cls.mode),
            structural_model=data.get("structural_model", cls.structural_model),
            reduction_ratio=float(data.get("reduction_ratio", cls.reduction_ratio)),
            pipeline_seed=int(seeds.get("pipeline", cls.pipeline_seed)),
            attribute_seed=int(seeds.get("attribute", cls.attribute_seed)),
            attribute=attribute_cfg,
            structure=structure_cfg,
            datasets=dataset_specs,
            results_dir=data.get("paths", {}).get("results_dir", cls.results_dir),
        )

    def dataset_spec(self, dataset: str) -> Optional[DatasetSpec]:
        """Return dataset-specific overrides if present."""

        return self.datasets.get(dataset)
