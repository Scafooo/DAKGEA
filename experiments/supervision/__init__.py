"""Supervision level experiments for testing data augmentation effectiveness.

This module provides tools for running experiments at various supervision levels (r%),
measuring how augmentation helps when training data is limited.

Usage:
    python -m experiments.supervision.runner config/experiments/supervision_example.yaml
"""

from .splitter import SupervisionExperimentSplitter, SupervisionSplit, SupervisionLevelData
from .config import SupervisionExperimentConfig
from .writer import SupervisionExperimentWriter
from .runner import SupervisionExperimentRunner

__all__ = [
    "SupervisionExperimentSplitter",
    "SupervisionSplit",
    "SupervisionLevelData",
    "SupervisionExperimentConfig",
    "SupervisionExperimentWriter",
    "SupervisionExperimentRunner",
]
