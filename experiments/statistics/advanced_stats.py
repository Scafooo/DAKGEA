#!/usr/bin/env python3
"""Advanced statistical analysis utilities."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, List
import math


def confidence_interval(values: List[float], confidence: float = 0.95) -> tuple[float, float] | None:
    """Calculate confidence interval for a list of values."""
    if len(values) < 2:
        return None

    n = len(values)
    m = mean(values)
    std_err = pstdev(values) / math.sqrt(n)

    # Use t-distribution critical value (approximation for large n)
    # For 95% confidence and large n, ~1.96
    if confidence == 0.95:
        t_critical = 1.96
    elif confidence == 0.99:
        t_critical = 2.576
    elif confidence == 0.90:
        t_critical = 1.645
    else:
        t_critical = 1.96

    margin = t_critical * std_err
    return (m - margin, m + margin)


def cohens_d(group1: List[float], group2: List[float]) -> float | None:
    """Calculate Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return None

    mean1 = mean(group1)
    mean2 = mean(group2)
    std1 = pstdev(group1)
    std2 = pstdev(group2)

    n1 = len(group1)
    n2 = len(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return None

    return (mean2 - mean1) / pooled_std


def paired_t_test(group1: List[float], group2: List[float]) -> Dict[str, float] | None:
    """
    Perform paired t-test (reduction vs augmentation on same dataset).
    Returns t-statistic and approximate p-value.
    """
    if len(group1) != len(group2) or len(group1) < 2:
        return None

    differences = [g2 - g1 for g1, g2 in zip(group1, group2)]
    n = len(differences)
    mean_diff = mean(differences)
    std_diff = pstdev(differences)

    if std_diff == 0:
        return None

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Approximate p-value (two-tailed)
    # For df > 30, use normal approximation
    df = n - 1
    if df > 30:
        # Normal approximation
        z = abs(t_stat)
        p_value = 2 * (1 - _normal_cdf(z))
    else:
        # Very rough approximation for small samples
        p_value = 0.05 if abs(t_stat) > 2.0 else 0.10

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "mean_difference": mean_diff,
    }


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF."""
    # Using error function approximation
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def percentiles(values: List[float], percentiles_list: List[float] = [25, 50, 75]) -> Dict[float, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {}

    sorted_values = sorted(values)
    n = len(sorted_values)
    result = {}

    for p in percentiles_list:
        if not (0 <= p <= 100):
            continue

        rank = (p / 100) * (n - 1)
        lower_idx = int(rank)
        upper_idx = min(lower_idx + 1, n - 1)
        weight = rank - lower_idx

        if lower_idx == upper_idx:
            result[p] = sorted_values[lower_idx]
        else:
            result[p] = sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight

    return result


def summarize_with_advanced_stats(values: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of values."""
    if not values:
        return {}

    stats = {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }

    # Add percentiles
    percs = percentiles(values, [25, 50, 75])
    stats.update({f"p{int(p)}": v for p, v in percs.items()})

    # Add confidence interval
    ci = confidence_interval(values)
    if ci:
        stats["ci_lower"], stats["ci_upper"] = ci

    return stats


__all__ = [
    "confidence_interval",
    "cohens_d",
    "paired_t_test",
    "percentiles",
    "summarize_with_advanced_stats",
]
