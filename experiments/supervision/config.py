"""Configuration for supervision level experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class SupervisionExperimentConfig:
    """Configuration for a supervision level experiment.

    Attributes:
        name: Experiment name
        dataset_name: Dataset identifier (e.g., "openea/D_W_15K_V1")
        pool_ratio: Fraction of alignments for pool (default 0.2)
        levels: List of supervision levels to test (e.g., [0.1, 0.2, ..., 1.0])
        augmentation_method: Augmentation method to use (e.g., "plm")
        augmentation_ratio: Augmentation ratio (e.g., 0.5 = +50% entities)
        models: List of alignment models to evaluate
        seed: Base random seed for reproducibility
        output_dir: Directory for experiment results
        writer: Dataset writer format (default "bert_int")
        resume: Whether to resume from cached results
    """
    name: str
    dataset_name: str
    pool_ratio: float = 0.2
    levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    augmentation_method: str = "plm"
    augmentation_ratio: float = 0.5
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    models: List[str] = field(default_factory=lambda: ["bert_int"])
    seed: int = 42
    output_dir: Optional[str] = None
    writer: str = "bert_int"
    resume: bool = True

    def __post_init__(self):
        # Validate levels
        for level in self.levels:
            if level <= 0 or level > 1.0:
                raise ValueError(f"Invalid supervision level: {level}. Must be in (0, 1.0]")

        # Sort levels for consistent ordering
        self.levels = sorted(set(self.levels))

        # Set default output dir
        if self.output_dir is None:
            self.output_dir = f"results/supervision/{self.name}"

    @classmethod
    def from_yaml(cls, path: Path) -> "SupervisionExperimentConfig":
        """Load configuration from YAML file."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract supervision_experiment section
        config_data = data.get("supervision_experiment", data)
        return cls(**config_data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "supervision_experiment": {
                "name": self.name,
                "dataset_name": self.dataset_name,
                "pool_ratio": self.pool_ratio,
                "levels": self.levels,
                "augmentation_method": self.augmentation_method,
                "augmentation_ratio": self.augmentation_ratio,
                "augmentation_config": self.augmentation_config,
                "models": self.models,
                "seed": self.seed,
                "output_dir": self.output_dir,
                "writer": self.writer,
                "resume": self.resume,
            }
        }

        with path.open("w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_level_tag(self, level: float) -> str:
        """Get string tag for a supervision level (e.g., '10pct' for 0.1)."""
        return f"{int(level * 100)}pct"

    def get_level_dir(self, level: float) -> Path:
        """Get output directory for a specific supervision level."""
        return Path(self.output_dir) / f"level_{self.get_level_tag(level)}"


__all__ = ["SupervisionExperimentConfig"]
