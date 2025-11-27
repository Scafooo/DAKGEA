#!/usr/bin/env python3
"""Outlier detection utilities for statistical analysis."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy import stats as scipy_stats


def detect_outliers_iqr(
    values: List[float],
    multiplier: float = 1.5,
) -> Tuple[List[int], List[int]]:
    """Detect outliers using Interquartile Range (IQR) method.

    Args:
        values: List of numeric values
        multiplier: IQR multiplier (typically 1.5 for outliers, 3.0 for extreme outliers)

    Returns:
        Tuple of (outlier_indices, extreme_outlier_indices)
    """
    if not values or len(values) < 4:
        return [], []

    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_indices = [i for i, v in enumerate(values) if v < lower_bound or v > upper_bound]

    # Extreme outliers (3 * IQR)
    extreme_lower = q1 - 3.0 * iqr
    extreme_upper = q3 + 3.0 * iqr
    extreme_outlier_indices = [i for i, v in enumerate(values) if v < extreme_lower or v > extreme_upper]

    return outlier_indices, extreme_outlier_indices


def detect_outliers_zscore(
    values: List[float],
    threshold: float = 3.0,
) -> List[int]:
    """Detect outliers using Z-score method.

    Args:
        values: List of numeric values
        threshold: Z-score threshold (typically 3.0 for 99.7% confidence)

    Returns:
        List of outlier indices
    """
    if not values or len(values) < 3:
        return []

    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return []

    z_scores = np.abs((arr - mean) / std)
    outlier_indices = [i for i, z in enumerate(z_scores) if z > threshold]

    return outlier_indices


def detect_outliers(
    values: List[float],
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> Tuple[List[int], List[int]]:
    """Detect outliers using specified method.

    Args:
        values: List of numeric values
        method: Detection method ("iqr" or "zscore")
        iqr_multiplier: IQR multiplier for IQR method
        zscore_threshold: Z-score threshold for Z-score method

    Returns:
        Tuple of (outlier_indices, extreme_outlier_indices)
        For zscore method, extreme_outlier_indices will be empty
    """
    if method == "iqr":
        return detect_outliers_iqr(values, iqr_multiplier)
    elif method == "zscore":
        outliers = detect_outliers_zscore(values, zscore_threshold)
        return outliers, []
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def get_outlier_summary(
    values: List[float],
    outlier_indices: List[int],
    extreme_outlier_indices: List[int],
) -> dict:
    """Get summary statistics about outliers.

    Args:
        values: Original list of values
        outlier_indices: Indices of outliers
        extreme_outlier_indices: Indices of extreme outliers

    Returns:
        Dictionary with outlier summary statistics
    """
    if not values:
        return {
            "total": 0,
            "outliers": 0,
            "extreme_outliers": 0,
            "outlier_percentage": 0.0,
            "extreme_outlier_percentage": 0.0,
        }

    arr = np.array(values)
    outlier_values = [values[i] for i in outlier_indices]
    extreme_outlier_values = [values[i] for i in extreme_outlier_indices]

    return {
        "total": len(values),
        "outliers": len(outlier_indices),
        "extreme_outliers": len(extreme_outlier_indices),
        "outlier_percentage": (len(outlier_indices) / len(values)) * 100,
        "extreme_outlier_percentage": (len(extreme_outlier_indices) / len(values)) * 100,
        "outlier_values": outlier_values,
        "extreme_outlier_values": extreme_outlier_values,
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


__all__ = [
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers",
    "get_outlier_summary",
]
