"""Public interface for the RREA (Relational Reflection Entity Alignment) alignment model package."""

from .config import DEFAULT_CONFIG, load_rrea_config, normalize_rrea_config
from .data_loader import load_rrea_data, RREADataBundle

__all__ = (
    "DEFAULT_CONFIG",
    "load_rrea_config",
    "normalize_rrea_config",
    "load_rrea_data",
    "RREADataBundle",
)
