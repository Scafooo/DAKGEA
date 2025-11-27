#!/usr/bin/env python3
"""Best/worst experiment tracking utilities."""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
from collections import defaultdict


def identify_best_worst_experiments(
    ratio_entries: Dict[str, List[Dict]],
    criteria: List[Dict[str, Any]],
    stage: str = "augmentation",
) -> Dict[str, Dict[str, List[Dict]]]:
    """Identify best and worst experiments per dataset based on multiple criteria.

    Args:
        ratio_entries: Dictionary of dataset -> list of experiment entries
        criteria: List of criteria dicts with 'metric' and 'maximize' keys
        stage: Stage to evaluate ("reduction" or "augmentation")

    Returns:
        Dictionary of dataset -> {"best": [...], "worst": [...]}
    """
    results = {}

    for dataset, entries in ratio_entries.items():
        best_experiments = []
        worst_experiments = []

        for criterion in criteria:
            metric = criterion["metric"]
            maximize = criterion.get("maximize", True)

            # Extract metric values from entries
            valid_entries = []
            for entry in entries:
                stage_data = entry.get(stage)
                if stage_data and metric in stage_data.get("metrics", {}):
                    valid_entries.append({
                        "entry": entry,
                        "value": stage_data["metrics"][metric],
                        "experiment": entry.get("experiment"),
                    })

            if not valid_entries:
                continue

            # Sort by metric value
            sorted_entries = sorted(
                valid_entries,
                key=lambda x: x["value"],
                reverse=maximize,
            )

            # Best
            if sorted_entries:
                best = sorted_entries[0]
                best_experiments.append({
                    "experiment": best["experiment"],
                    "metric": metric,
                    "value": best["value"],
                    "maximize": maximize,
                })

            # Worst
            if sorted_entries:
                worst = sorted_entries[-1]
                worst_experiments.append({
                    "experiment": worst["experiment"],
                    "metric": metric,
                    "value": worst["value"],
                    "maximize": maximize,
                })

        results[dataset] = {
            "best": best_experiments,
            "worst": worst_experiments,
        }

    return results


def get_top_n_experiments(
    ratio_entries: Dict[str, List[Dict]],
    metric: str,
    n: int = 5,
    stage: str = "augmentation",
    maximize: bool = True,
) -> Dict[str, List[Dict]]:
    """Get top N experiments per dataset for a specific metric.

    Args:
        ratio_entries: Dictionary of dataset -> list of experiment entries
        metric: Metric to rank by
        n: Number of top experiments to return
        stage: Stage to evaluate ("reduction" or "augmentation")
        maximize: Whether higher values are better

    Returns:
        Dictionary of dataset -> list of top N experiments
    """
    results = {}

    for dataset, entries in ratio_entries.items():
        # Extract metric values
        valid_entries = []
        for entry in entries:
            stage_data = entry.get(stage)
            if stage_data and metric in stage_data.get("metrics", {}):
                valid_entries.append({
                    "experiment": entry.get("experiment"),
                    "value": stage_data["metrics"][metric],
                    "ratio": stage_data.get("ratio"),
                    "all_metrics": stage_data.get("metrics", {}),
                })

        if not valid_entries:
            results[dataset] = []
            continue

        # Sort and take top N
        sorted_entries = sorted(
            valid_entries,
            key=lambda x: x["value"],
            reverse=maximize,
        )

        results[dataset] = sorted_entries[:n]

    return results


def calculate_improvement_rankings(
    ratio_entries: Dict[str, List[Dict]],
    metric: str = "hits@1",
) -> Dict[str, List[Dict]]:
    """Calculate improvement rankings (augmentation - reduction).

    Args:
        ratio_entries: Dictionary of dataset -> list of experiment entries
        metric: Metric to calculate improvement for

    Returns:
        Dictionary of dataset -> list of experiments ranked by improvement
    """
    results = {}

    for dataset, entries in ratio_entries.items():
        improvements = []

        for entry in entries:
            red_data = entry.get("reduction")
            aug_data = entry.get("augmentation")

            if not red_data or not aug_data:
                continue

            red_value = red_data.get("metrics", {}).get(metric)
            aug_value = aug_data.get("metrics", {}).get(metric)

            if red_value is None or aug_value is None:
                continue

            delta = aug_value - red_value
            delta_percentage = (delta / red_value * 100) if red_value != 0 else 0

            improvements.append({
                "experiment": entry.get("experiment"),
                "reduction_value": red_value,
                "augmentation_value": aug_value,
                "delta": delta,
                "delta_percentage": delta_percentage,
                "reduction_ratio": red_data.get("ratio"),
                "augmentation_ratio": aug_data.get("ratio"),
            })

        # Sort by delta (improvement)
        improvements.sort(key=lambda x: x["delta"], reverse=True)
        results[dataset] = improvements

    return results


def get_summary_statistics(
    tracking_results: Dict[str, Dict[str, List[Dict]]],
) -> Dict[str, Dict]:
    """Get summary statistics for best/worst tracking results.

    Args:
        tracking_results: Results from identify_best_worst_experiments()

    Returns:
        Summary statistics per dataset
    """
    summary = {}

    for dataset, results in tracking_results.items():
        best = results.get("best", [])
        worst = results.get("worst", [])

        # Group by metric
        best_by_metric = defaultdict(list)
        worst_by_metric = defaultdict(list)

        for item in best:
            best_by_metric[item["metric"]].append(item["value"])

        for item in worst:
            worst_by_metric[item["metric"]].append(item["value"])

        summary[dataset] = {
            "num_best_tracked": len(best),
            "num_worst_tracked": len(worst),
            "best_by_metric": dict(best_by_metric),
            "worst_by_metric": dict(worst_by_metric),
        }

    return summary


__all__ = [
    "identify_best_worst_experiments",
    "get_top_n_experiments",
    "calculate_improvement_rankings",
    "get_summary_statistics",
]
