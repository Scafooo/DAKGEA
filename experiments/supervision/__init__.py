"""Supervision level experiments for testing data augmentation effectiveness.

This module provides tools for running experiments at various supervision levels (r%),
measuring how augmentation helps when training data is limited.

Usage:
    # Using bash script (recommended)
    ./scripts/run_supervision_experiment.sh supervision_dw15k

    # Or directly with Python
    python -m experiments.supervision.run_supervision config/experiments/supervision_dw15k.yaml
"""

from .splitter import SupervisionExperimentSplitter, SupervisionSplit, SupervisionLevelData
from .config import SupervisionExperimentConfig
from .writer import SupervisionExperimentWriter
from .run_supervision import SupervisionExperimentOrchestrator

__all__ = [
    "SupervisionExperimentSplitter",
    "SupervisionSplit",
    "SupervisionLevelData",
    "SupervisionExperimentConfig",
    "SupervisionExperimentWriter",
    "SupervisionExperimentOrchestrator",
]
