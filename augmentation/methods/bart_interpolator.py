"""Legacy access to PLM interpolation utilities."""

from src.augmentation.methods.bart_interpolator import (
    BartInterpolatorPLM,
    PairExample,
    _clean_pred,
)

__all__ = ["BartInterpolatorPLM", "PairExample", "_clean_pred"]
