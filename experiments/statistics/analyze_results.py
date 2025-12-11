#!/usr/bin/env python3
"""Compare reduction vs augmentation metrics aggregated per dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.loader import PROJECT_ROOT, load_yaml
from experiments.statistics.config import get_statistics_config
from experiments.statistics.advanced_stats import (
    summarize_with_advanced_stats,
    cohens_d,
    paired_t_test,
)
from experiments.statistics.visualizations import (
    plot_heatmap,
    plot_boxplot,
    plot_violin,
    plot_scatter_correlation,
    plot_delta_chart,
    plot_radar_chart,
    plot_ridge,
    plot_performance_matrix,
    plot_surface_3d,
    plot_interactive_surface_3d,
)
from experiments.statistics.console_output import (
    create_console,
    create_progress_bar,
    print_header,
    print_dataset_header,
    format_delta_with_color,
    print_metric_summary,
    create_summary_table,
    RICH_AVAILABLE,
)
from experiments.statistics.outlier_detection import detect_outliers, get_outlier_summary
from experiments.statistics.tracking import (
    identify_best_worst_experiments,
    get_top_n_experiments,
    calculate_improvement_rankings,
)
from experiments.statistics.latex_document import create_results_document
from experiments.statistics.exporters import (
    write_dataset_summary_latex,
    write_comparison_tables_latex,
    write_detailed_comparison_tables_latex,
)
from src.logger import get_logger

# Load configuration
stats_config = get_statistics_config()

# Configuration constants (loaded from YAML)
RESULTS_FILENAME = stats_config.results_filename
SUMMARY_FILENAME = stats_config.summary_filename
METADATA_FILENAME = stats_config.metadata_filename
STAGES = tuple(stats_config.stages)
DEFAULT_METRICS = stats_config.default_metrics
PLOT_METRICS = stats_config.plot_metrics  # Dict[str, List[str]] - grouped by metric type
METRIC_GROUPS = stats_config.metric_groups  # Dict[str, List[str]] - all metric groups
METRIC_COLORS = stats_config.metric_colors
DEFAULT_DPI = stats_config.default_dpi
PLOT_DPI = DEFAULT_DPI

logger = get_logger("experiments.statistics")


def _parse_experiment_name(path: Path) -> Tuple[str | None, float | None, float | None]:
    parts = path.name.split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        reduction_raw = int(parts[-2])
        augmentation_raw = int(parts[-1])
    except ValueError:
        return None, None, None
    reduction_ratio = reduction_raw / 10.0
    augmentation_ratio = augmentation_raw / 10.0
    dataset = "_".join(parts[:-2]) or None
    return dataset, reduction_ratio, augmentation_ratio


def _sanitize_metric(metric: str, value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    # Clamp metrics that should be in [0, 1] range (from configuration)
    normalized_metrics = set(stats_config.normalized_metrics)
    if metric.lower() in normalized_metrics:
        return max(0.0, min(1.0, value))
    return value


def parse_args(default_plots_dir: Path, default_results_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reduction vs augmentation metrics per dataset.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Experiment directories to include. If omitted, scans all ./results/* directories.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to aggregate (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Restrict aggregation to specific model names (default: use all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only include these dataset names (case-insensitive).",
    )
    parser.add_argument(
        "--reduction-ratios",
        nargs="+",
        type=float,
        help="Only include experiments whose reduction ratio matches one of these values.",
    )
    parser.add_argument(
        "--augmentation-ratios",
        nargs="+",
        type=float,
        help="Only include experiments whose augmentation ratio matches one of these values.",
    )
    parser.add_argument(
        "--plot-metric",
        default="hits@1",
        help="Metric used for the comparison plots (default: hits@1).",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to dump the aggregated statistics as JSON.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(default_plots_dir),
        help=f"Directory where comparison plots will be saved (default: {default_plots_dir}).",
    )
    parser.add_argument(
        "--tsv-dir",
        default=str(default_plots_dir),
        help=f"Directory where TSV summaries will be written (default: {default_plots_dir}).",
    )
    parser.add_argument(
        "--results-root",
        default=str(default_results_dir),
        help=f"Root directory containing experiment runs (default: {default_results_dir}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Resolution (DPI) for saved plots (default: %(default)s).",
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=["tsv", "latex", "latex-doc"],
        default=["tsv"],
        help="Export formats to generate (default: %(default)s). Options: tsv, latex (tables only), latex-doc (complete document).",
    )
    parser.add_argument(
        "--enable-advanced-plots",
        action="store_true",
        help="Generate heatmaps, boxplots, violin plots, scatter plots, and delta charts.",
    )
    parser.add_argument(
        "--advanced-stats",
        action="store_true",
        help="Include confidence intervals, t-tests, effect sizes in console output.",
    )
    return parser.parse_args()


def discover_experiments(candidate_paths: Iterable[str], results_root: Path) -> List[Path]:
    if candidate_paths:
        paths = [Path(p) for p in candidate_paths]
        return [p.resolve() for p in paths if p.exists()]
    root = results_root
    if not root.exists():
        raise FileNotFoundError("No experiment directories found and no explicit paths provided.")
    skip = set(stats_config.skip_directories)
    return sorted(
        p.resolve()
        for p in root.iterdir()
        if p.is_dir() and p.name not in skip
    )


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_metrics(
    stage_dir: Path,
    metrics: List[str],
    model_filter: List[str] | None,
) -> List[Dict]:
    result_file = stage_dir / RESULTS_FILENAME
    if not result_file.exists():
        return []
    payload = load_json(result_file)
    records: List[Dict] = []
    for model_name, stats in payload.items():
        if model_filter and model_name not in model_filter:
            continue
        record_metrics = {}
        for metric in metrics:
            if metric not in stats:
                continue
            value = _sanitize_metric(metric, stats[metric])
            if value is not None:
                record_metrics[metric] = value
        if record_metrics:
            records.append({"model": model_name, "metrics": record_metrics})
    return records


def load_stage_ratio(stage_dir: Path) -> float | None:
    summary = load_json(stage_dir / SUMMARY_FILENAME)
    ratio = summary.get("ratio")
    if isinstance(ratio, (int, float)):
        return float(ratio)
    return None


def average_record(records: List[Dict], metrics: List[str]) -> Dict[str, float]:
    """Average metric values across models for a single experiment."""
    averages: Dict[str, float] = {}
    for metric in metrics:
        values = [record["metrics"][metric] for record in records if metric in record["metrics"]]
        if values:
            averages[metric] = mean(values)
    return averages


def collect_data(
    experiments: List[Path],
    metrics: List[str],
    model_filter: List[str] | None,
    dataset_filter: set[str] | None,
    reduction_filter: set[float] | None,
    augmentation_filter: set[float] | None,
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, List[Dict]]]:
    datasets: Dict[str, Dict[str, Dict]] = defaultdict(
        lambda: {
            stage: {"records": [], "ratios": [], "ratio_records": defaultdict(list)}
            for stage in STAGES
        }
    )
    ratio_entries: Dict[str, List[Dict]] = defaultdict(list)
    for exp_dir in experiments:
        metadata = load_json(exp_dir / METADATA_FILENAME)
        parsed_dataset, parsed_reduction, parsed_augmentation = _parse_experiment_name(exp_dir)
        dataset_name = metadata.get("dataset") or parsed_dataset or exp_dir.name
        dataset_key = dataset_name.lower()
        if dataset_filter and dataset_key not in dataset_filter:
            continue

        reduction_ratio_meta = metadata.get("ratio")
        try:
            reduction_ratio_value = float(reduction_ratio_meta)
        except (TypeError, ValueError):
            reduction_ratio_value = parsed_reduction
        reduction_ratio_key = round(reduction_ratio_value, 6) if reduction_ratio_value is not None else None
        if reduction_filter and (reduction_ratio_key is None or reduction_ratio_key not in reduction_filter):
            continue

        entry_data: Dict[str, Dict] = {"experiment": exp_dir.name}
        for stage in STAGES:
            stage_dir = exp_dir / stage
            if not stage_dir.exists():
                continue
            records = load_stage_metrics(stage_dir, metrics, model_filter)
            if not records:
                continue
            ratio = load_stage_ratio(stage_dir)
            if ratio is None:
                ratio = reduction_ratio_value if stage == "reduction" else parsed_augmentation
            elif stage == "augmentation" and parsed_augmentation is not None:
                ratio = parsed_augmentation
            ratio_key = round(float(ratio), 6) if ratio is not None else None
            if stage == "augmentation" and augmentation_filter:
                if ratio_key is None or ratio_key not in augmentation_filter:
                    continue

            stage_bucket = datasets[dataset_name][stage]
            stage_bucket["records"].extend(records)
            if ratio_key is not None:
                stage_bucket["ratios"].append(ratio)
                stage_bucket["ratio_records"][ratio_key].extend(records)
            entry_data[stage] = {
                "ratio": ratio,
                "metrics": average_record(records, metrics),
            }
        if entry_data:
            ratio_entries[dataset_name].append(entry_data)
    return datasets, ratio_entries


def summarize_records(records: List[Dict], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        values = [record["metrics"][metric] for record in records if metric in record["metrics"]]
        if not values:
            continue
        summary[metric] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def summarize_ratios(ratios: List[float]) -> Dict[str, float] | None:
    if not ratios:
        return None
    return {
        "count": len(ratios),
        "mean": mean(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_dataset(
    dataset: str,
    metric: str,
    stage_stats: Dict[str, Dict[str, Dict]],
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]] = None,
) -> None:
    """Plot bar chart comparing reduction vs augmentation for a single metric.

    Args:
        dataset: Dataset name
        metric: Metric to plot
        stage_stats: Stage statistics dictionary
        plots_dir: Output directory for plots
        stage_colors: Color configuration for stages
    """
    # Default colors if not provided
    if stage_colors is None:
        stage_colors = {
            "reduction": {"primary": "#264653"},
            "augmentation": {"primary": "#e76f51"},
        }

    values = []
    labels = []
    colors = []
    for stage in STAGES:
        stats = stage_stats.get(stage, {})
        metric_stats = stats.get(metric)
        if metric_stats:
            values.append(metric_stats["mean"])
            labels.append(stage.capitalize())
            colors.append(stage_colors.get(stage, {}).get("primary", "#264653"))

    if len(values) < 2:
        return

    ensure_dir(plots_dir)
    fig_size = stats_config.figure_size_default
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"{value:.3f}", ha="center", va="bottom", fontsize=10, fontweight='bold')

    # Calculate and display delta
    if len(values) == 2:
        delta = values[1] - values[0]  # augmentation - reduction
        delta_pct = (delta / values[0] * 100) if values[0] != 0 else 0
        delta_color = "#2a9d8f" if delta > 0 else "#e63946"
        ax.text(0.5, 0.95, f"Δ = {delta:+.4f} ({delta_pct:+.2f}%)",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold', color=delta_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=delta_color, linewidth=2, alpha=0.9))

    ax.set_title(f"{dataset} – {metric}", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / f"{dataset}_{metric.replace('@', 'at').replace('/', '_')}.png"
    plt.savefig(outfile, dpi=PLOT_DPI)
    plt.close(fig)


def _aggregate_ratio_metrics(entries: List[Dict], metrics: List[str]) -> Dict[float, Dict[str, float]]:
    best_per_ratio: Dict[float, Dict[str, float]] = {}
    best_score: Dict[float, float] = {}
    for entry in entries:
        aug = entry.get("augmentation")
        if not aug:
            continue
        try:
            ratio_key = round(float(aug.get("ratio")), 6)
        except (TypeError, ValueError):
            continue
        score = aug["metrics"].get("hits@1")
        if score is None:
            continue
        if ratio_key not in best_score or score > best_score[ratio_key]:
            best_score[ratio_key] = score
            best_per_ratio[ratio_key] = {metric: aug["metrics"].get(metric) for metric in metrics}
    return best_per_ratio


def build_ratio_plot_data(
    ratio_entries: Dict[str, List[Dict]],
    metrics: List[str],
) -> Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]]:
    plot_data: Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]] = {}
    for dataset, entries in ratio_entries.items():
        logger.debug(f"[BUILD_RATIO_PLOT] Processing dataset={dataset} with {len(entries)} entries")
        dataset_block = plot_data.setdefault(dataset, {})
        best_per_ratio: Dict[Tuple[float, float], Dict] = {}
        entries_with_both = 0
        entries_red_only = 0
        entries_aug_only = 0
        for entry in entries:
            red = entry.get("reduction")
            aug = entry.get("augmentation")
            if not red or not aug:
                if red and not aug:
                    entries_red_only += 1
                elif aug and not red:
                    entries_aug_only += 1
                continue
            entries_with_both += 1
            try:
                red_ratio = round(float(red.get("ratio")), 6)
                aug_ratio = round(float(aug.get("ratio")), 6)
            except (TypeError, ValueError):
                continue
            pair = (red_ratio, aug_ratio)
            current_best = best_per_ratio.get(pair)
            current_score = current_best["augmentation"]["metrics"].get("hits@1") if current_best else None
            candidate_score = aug["metrics"].get("hits@1")
            if current_best is None or (candidate_score is not None and candidate_score > (current_score or -1)):
                best_per_ratio[pair] = entry

        logger.debug(f"[BUILD_RATIO_PLOT]   Entries: {entries_with_both} with both, {entries_red_only} reduction-only, {entries_aug_only} augmentation-only")
        logger.debug(f"[BUILD_RATIO_PLOT]   Best pairs found: {len(best_per_ratio)}")

        for (red_ratio, aug_ratio), entry in best_per_ratio.items():
            red_metrics = entry["reduction"]["metrics"]
            aug_metrics = entry["augmentation"]["metrics"]
            red_block = dataset_block.setdefault(red_ratio, {})
            red_block[aug_ratio] = {
                "reduction": {metric: red_metrics.get(metric) for metric in metrics},
                "augmentation": {metric: aug_metrics.get(metric) for metric in metrics},
            }

        logger.debug(f"[BUILD_RATIO_PLOT]   Final plot_data for {dataset}: {len(dataset_block)} reduction ratios")

    return plot_data


def plot_ratio_groups(
    dataset: str,
    ratio_plot_data: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    plots_dir: Path,
) -> None:
    ratio_dir = plots_dir / "ratio_charts"
    ensure_dir(ratio_dir)
    for red_ratio, aug_group in sorted(ratio_plot_data.items()):
        if not aug_group:
            continue
        aug_ratios = sorted(aug_group.keys())
        if not aug_ratios:
            continue

        metrics_count = len(metrics)
        bar_width = 0.35
        gap_between_metrics = 0.15
        gap_between_ratios = bar_width * metrics_count * 2 + 0.8

        fig, ax = plt.subplots(
            figsize=(max(8, len(aug_ratios) * metrics_count), 5)
        )
        positions = []
        labels = []
        current_x = 0.0
        axis_transform = ax.get_xaxis_transform()

        for aug_ratio in aug_ratios:
            data = aug_group[aug_ratio]
            for metric_idx, metric in enumerate(metrics):
                red_value = data["reduction"].get(metric)
                aug_value = data["augmentation"].get(metric)
                metric_offset = metric_idx * (2 * bar_width + gap_between_metrics)
                red_x = current_x + metric_offset
                aug_x = red_x + bar_width
                if red_value is not None:
                    ax.bar(
                        red_x,
                        red_value,
                        width=bar_width,
                        color="#264653",
                        label="Reduction" if (aug_ratio == aug_ratios[0] and metric_idx == 0) else "",
                    )
                    ax.text(red_x, red_value + 0.01, f"{red_value:.2f}", ha="center", va="bottom", fontsize=7)
                if aug_value is not None:
                    ax.bar(
                        aug_x,
                        aug_value,
                        width=bar_width,
                        color="#e76f51",
                        label="Augmentation" if (aug_ratio == aug_ratios[0] and metric_idx == 0) else "",
                    )
                    ax.text(aug_x, aug_value + 0.01, f"{aug_value:.2f}", ha="center", va="bottom", fontsize=7)
                center_x = red_x + bar_width / 2
                ax.text(
                    center_x,
                    -0.07,
                    metric,
                    ha="right",
                    va="top",
                    fontsize=8,
                    rotation=45,
                    transform=axis_transform,
                )

            ratio_center = current_x + (metrics_count * (2 * bar_width + gap_between_metrics) - gap_between_metrics) / 2
            positions.append(ratio_center)
            labels.append(f"{aug_ratio:.2f}")
            current_x += metrics_count * (2 * bar_width + gap_between_metrics) + gap_between_ratios

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Augmentation ratio")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dataset} – reduction ratio {red_ratio:.2f}")
        ax.legend()
        fig.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        outfile = ratio_dir / f"{dataset}_red{red_ratio:.2f}_comparison.png"
        plt.savefig(outfile, dpi=PLOT_DPI)
        plt.close(fig)


def plot_ratio_trends(
    dataset: str,
    ratio_plot_data: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    plots_dir: Path,
) -> None:
    trend_dir = plots_dir / "ratio_trends"
    ensure_dir(trend_dir)

    # Debug logging
    logger.debug(f"[RATIO_TRENDS] Called for dataset={dataset}")
    logger.debug(f"[RATIO_TRENDS]   ratio_plot_data has {len(ratio_plot_data)} reduction ratios")
    logger.debug(f"[RATIO_TRENDS]   plots_dir={plots_dir}")
    logger.debug(f"[RATIO_TRENDS]   trend_dir={trend_dir}")

    stage_styles = {
        "reduction": {"linestyle": "--", "marker": "x", "alpha": 0.9},
        "augmentation": {"linestyle": "-", "marker": "o", "alpha": 0.9},
    }

    plots_generated = 0
    for red_ratio, aug_group in sorted(ratio_plot_data.items()):
        logger.debug(f"[RATIO_TRENDS]   Processing red_ratio={red_ratio}, aug_group has {len(aug_group)} entries")
        if not aug_group:
            logger.debug(f"[RATIO_TRENDS]   Skipping red_ratio={red_ratio}: aug_group is empty")
            continue
        aug_ratios = sorted(aug_group.keys())
        if not aug_ratios:
            logger.debug(f"[RATIO_TRENDS]   Skipping red_ratio={red_ratio}: no aug_ratios")
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(aug_ratios) * 1.2), 4.5))
        for metric in metrics:
            color = METRIC_COLORS.get(metric, None)
            if color is None:
                color = "#333333"
            for stage in ("reduction", "augmentation"):
                y_values = [
                    aug_group[ratio][stage].get(metric) if stage in aug_group[ratio] else None
                    for ratio in aug_ratios
                ]
                if not any(value is not None for value in y_values):
                    continue
                y_series = [value if value is not None else float("nan") for value in y_values]
                style = stage_styles.get(stage, {})
                ax.plot(
                    aug_ratios,
                    y_series,
                    label=f"{metric} ({stage})",
                    color=color,
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker"),
                    alpha=style.get("alpha", 1.0),
                )
        ax.set_xlabel("Augmentation ratio")
        ax.set_ylabel("Score")
        ax.set_title(f"{dataset} – trend (reduction {red_ratio:.2f})")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        outfile = trend_dir / f"{dataset}_red{red_ratio:.2f}_trend.png"
        plt.savefig(outfile, dpi=PLOT_DPI)
        plt.close(fig)
        plots_generated += 1
        logger.debug(f"[RATIO_TRENDS]   ✓ Generated plot: {outfile}")

    if plots_generated > 0:
        logger.info(f"Generated {plots_generated} ratio trend plots in {trend_dir}")
    else:
        logger.warning(f"No ratio trend plots generated for dataset={dataset} (ratio_plot_data was empty or invalid)")


def build_reverse_ratio_plot_data(
    ratio_entries: Dict[str, List[Dict]],
    metrics: List[str],
) -> Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]]:
    """Build plot data with augmentation ratio as primary key, reduction ratio as secondary.

    This is the reverse of build_ratio_plot_data - used for plotting reduction trends.

    Returns:
        Dict[dataset -> augmentation_ratio -> reduction_ratio -> stage -> metric -> value]
    """
    plot_data: Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]] = {}
    for dataset, entries in ratio_entries.items():
        logger.debug(f"[BUILD_REVERSE_RATIO_PLOT] Processing dataset={dataset} with {len(entries)} entries")
        dataset_block = plot_data.setdefault(dataset, {})
        best_per_ratio: Dict[Tuple[float, float], Dict] = {}
        entries_with_both = 0
        entries_red_only = 0
        entries_aug_only = 0
        for entry in entries:
            red = entry.get("reduction")
            aug = entry.get("augmentation")
            if not red or not aug:
                if red and not aug:
                    entries_red_only += 1
                elif aug and not red:
                    entries_aug_only += 1
                continue
            entries_with_both += 1
            try:
                red_ratio = round(float(red.get("ratio")), 6)
                aug_ratio = round(float(aug.get("ratio")), 6)
            except (TypeError, ValueError):
                continue
            pair = (aug_ratio, red_ratio)  # Reversed compared to build_ratio_plot_data
            current_best = best_per_ratio.get(pair)
            current_score = current_best["augmentation"]["metrics"].get("hits@1") if current_best else None
            candidate_score = aug["metrics"].get("hits@1")
            if current_best is None or (candidate_score is not None and candidate_score > (current_score or -1)):
                best_per_ratio[pair] = entry

        logger.debug(f"[BUILD_REVERSE_RATIO_PLOT]   Entries: {entries_with_both} with both, {entries_red_only} reduction-only, {entries_aug_only} augmentation-only")
        logger.debug(f"[BUILD_REVERSE_RATIO_PLOT]   Best pairs found: {len(best_per_ratio)}")

        for (aug_ratio, red_ratio), entry in best_per_ratio.items():
            red_metrics = entry["reduction"]["metrics"]
            aug_metrics = entry["augmentation"]["metrics"]
            aug_block = dataset_block.setdefault(aug_ratio, {})  # Primary key is augmentation
            aug_block[red_ratio] = {  # Secondary key is reduction
                "reduction": {metric: red_metrics.get(metric) for metric in metrics},
                "augmentation": {metric: aug_metrics.get(metric) for metric in metrics},
            }

        logger.debug(f"[BUILD_REVERSE_RATIO_PLOT]   Final plot_data for {dataset}: {len(dataset_block)} augmentation ratios")

    return plot_data


def plot_reduction_trends(
    dataset: str,
    ratio_plot_data: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    plots_dir: Path,
) -> None:
    """Plot trends with reduction ratio on X-axis (augmentation ratio fixed).

    This is the inverse of plot_ratio_trends - shows how metrics change as reduction varies.

    Args:
        dataset: Dataset name
        ratio_plot_data: Dict[aug_ratio -> red_ratio -> stage -> metric -> value]
        metrics: List of metrics to plot
        plots_dir: Base output directory
    """
    trend_dir = plots_dir / "reduction_trends"
    ensure_dir(trend_dir)

    # Debug logging
    logger.debug(f"[REDUCTION_TRENDS] Called for dataset={dataset}")
    logger.debug(f"[REDUCTION_TRENDS]   ratio_plot_data has {len(ratio_plot_data)} augmentation ratios")
    logger.debug(f"[REDUCTION_TRENDS]   plots_dir={plots_dir}")
    logger.debug(f"[REDUCTION_TRENDS]   trend_dir={trend_dir}")

    stage_styles = {
        "reduction": {"linestyle": "--", "marker": "x", "alpha": 0.9},
        "augmentation": {"linestyle": "-", "marker": "o", "alpha": 0.9},
    }

    plots_generated = 0
    for aug_ratio, red_group in sorted(ratio_plot_data.items()):
        logger.debug(f"[REDUCTION_TRENDS]   Processing aug_ratio={aug_ratio}, red_group has {len(red_group)} entries")
        if not red_group:
            logger.debug(f"[REDUCTION_TRENDS]   Skipping aug_ratio={aug_ratio}: red_group is empty")
            continue
        red_ratios = sorted(red_group.keys())
        if not red_ratios:
            logger.debug(f"[REDUCTION_TRENDS]   Skipping aug_ratio={aug_ratio}: no red_ratios")
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(red_ratios) * 1.2), 4.5))
        for metric in metrics:
            color = METRIC_COLORS.get(metric, None)
            if color is None:
                color = "#333333"
            for stage in ("reduction", "augmentation"):
                y_values = [
                    red_group[ratio][stage].get(metric) if stage in red_group[ratio] else None
                    for ratio in red_ratios
                ]
                if not any(value is not None for value in y_values):
                    continue
                y_series = [value if value is not None else float("nan") for value in y_values]
                style = stage_styles.get(stage, {})
                ax.plot(
                    red_ratios,
                    y_series,
                    label=f"{metric} ({stage})",
                    color=color,
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker"),
                    alpha=style.get("alpha", 1.0),
                )
        ax.set_xlabel("Reduction ratio")
        ax.set_ylabel("Score")
        ax.set_title(f"{dataset} – trend (augmentation {aug_ratio:.2f})")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        outfile = trend_dir / f"{dataset}_aug{aug_ratio:.2f}_reduction_trend.png"
        plt.savefig(outfile, dpi=PLOT_DPI)
        plt.close(fig)
        plots_generated += 1
        logger.debug(f"[REDUCTION_TRENDS]   ✓ Generated plot: {outfile}")

    if plots_generated > 0:
        logger.info(f"Generated {plots_generated} reduction trend plots in {trend_dir}")
    else:
        logger.warning(f"No reduction trend plots generated for dataset={dataset} (ratio_plot_data was empty or invalid)")


def write_dataset_summary_tsv(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
) -> None:
    """Write enhanced TSV export with min, max, and delta percentage."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        header = [
            "dataset",
            "metric",
            "reduction_mean",
            "reduction_std",
            "reduction_min",
            "reduction_max",
            "reduction_count",
            "augmentation_mean",
            "augmentation_std",
            "augmentation_min",
            "augmentation_max",
            "augmentation_count",
            "delta_mean",
            "delta_percentage",
        ]
        handle.write("\t".join(header) + "\n")
        for dataset, stage_stats in sorted(dataset_stage_stats.items()):
            for metric in metrics:
                red_stats = stage_stats.get("reduction", {}).get(metric)
                aug_stats = stage_stats.get("augmentation", {}).get(metric)
                if not red_stats and not aug_stats:
                    continue

                delta = None
                delta_pct = None
                if red_stats and aug_stats:
                    delta = aug_stats["mean"] - red_stats["mean"]
                    if red_stats["mean"] != 0:
                        delta_pct = (delta / red_stats["mean"]) * 100

                row = [
                    dataset,
                    metric,
                    f"{red_stats['mean']:.6f}" if red_stats else "",
                    f"{red_stats['std']:.6f}" if red_stats else "",
                    f"{red_stats['min']:.6f}" if red_stats else "",
                    f"{red_stats['max']:.6f}" if red_stats else "",
                    str(red_stats["count"]) if red_stats else "",
                    f"{aug_stats['mean']:.6f}" if aug_stats else "",
                    f"{aug_stats['std']:.6f}" if aug_stats else "",
                    f"{aug_stats['min']:.6f}" if aug_stats else "",
                    f"{aug_stats['max']:.6f}" if aug_stats else "",
                    str(aug_stats["count"]) if aug_stats else "",
                    f"{delta:.6f}" if delta is not None else "",
                    f"{delta_pct:.2f}" if delta_pct is not None else "",
                ]
                handle.write("\t".join(row) + "\n")


def write_ratio_summary_tsv(
    path: Path,
    ratio_entries: Dict[str, List[Dict]],
    metrics: List[str],
) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        header = [
            "dataset",
            "reduction_ratio",
            "augmentation_ratio",
            "metric",
            "reduction_mean",
            "augmentation_mean",
            "delta_mean",
            "experiments",
        ]
        handle.write("\t".join(header) + "\n")
        for dataset, entries in sorted(ratio_entries.items()):
            group: Dict[Tuple[float | None, float | None], List[Dict]] = defaultdict(list)
            for entry in entries:
                red_ratio = entry.get("reduction", {}).get("ratio")
                aug_ratio = entry.get("augmentation", {}).get("ratio")
                if red_ratio is None or aug_ratio is None:
                    continue
                key = (round(float(red_ratio), 6), round(float(aug_ratio), 6))
                group[key].append(entry)
            for (red_ratio, aug_ratio), grouped_entries in sorted(group.items()):
                for metric in metrics:
                    red_values = [
                        e.get("reduction", {}).get("metrics", {}).get(metric)
                        for e in grouped_entries
                        if metric in e.get("reduction", {}).get("metrics", {})
                    ]
                    aug_values = [
                        e.get("augmentation", {}).get("metrics", {}).get(metric)
                        for e in grouped_entries
                        if metric in e.get("augmentation", {}).get("metrics", {})
                    ]
                    if not red_values and not aug_values:
                        continue
                    red_mean = mean(red_values) if red_values else None
                    aug_mean = mean(aug_values) if aug_values else None
                    delta = (
                        aug_mean - red_mean
                        if (red_mean is not None and aug_mean is not None)
                        else None
                    )
                    row = [
                        dataset,
                        "" if red_ratio is None else f"{red_ratio:.6f}",
                        "" if aug_ratio is None else f"{aug_ratio:.6f}",
                        metric,
                        "" if red_mean is None else f"{red_mean:.6f}",
                        "" if aug_mean is None else f"{aug_mean:.6f}",
                        "" if delta is None else f"{delta:.6f}",
                        str(len(grouped_entries)),
                    ]
                    handle.write("\t".join(row) + "\n")


def load_global_config() -> Dict:
    cfg_path = PROJECT_ROOT / "config/global.yaml"
    return load_yaml(cfg_path) if cfg_path.exists() else {}


def main() -> None:
    logger = get_logger("experiments.statistics", level="INFO")

    # Use statistics config for defaults, fall back to global config
    global_cfg = load_global_config()
    default_plots_dir = (
        PROJECT_ROOT /
        global_cfg.get("paths", {}).get("statistics", stats_config.plots_output_dir)
    ).resolve()
    default_results_dir = (
        PROJECT_ROOT /
        global_cfg.get("paths", {}).get("results", "results")
    ).resolve()
    args = parse_args(default_plots_dir, default_results_dir)
    global PLOT_DPI
    PLOT_DPI = args.dpi
    experiments = discover_experiments(args.paths, Path(args.results_root).resolve())
    dataset_filter = {name.lower() for name in args.datasets} if args.datasets else None
    reduction_filter = {round(val, 6) for val in args.reduction_ratios} if args.reduction_ratios else None
    augmentation_filter = {round(val, 6) for val in args.augmentation_ratios} if args.augmentation_ratios else None

    datasets, ratio_entries = collect_data(
        experiments,
        args.metrics,
        args.models,
        dataset_filter,
        reduction_filter,
        augmentation_filter,
    )
    if not datasets:
        raise SystemExit("No metrics found in the specified experiments.")

    aggregated_output = {}
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    plots_base = Path(args.plots_dir)
    # Ensure plots_base is absolute - resolve relative to PROJECT_ROOT if needed
    if not plots_base.is_absolute():
        plots_base = (PROJECT_ROOT / plots_base).resolve()
    else:
        plots_base = plots_base.resolve()
    logger.info("Loaded data from %d experiment directories.", len(experiments))
    logger.info("")
    for dataset, stages in sorted(datasets.items()):
        logger.info("=== Dataset: %s ===", dataset)
        dataset_entries = ratio_entries.get(dataset, [])
        stage_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for stage in STAGES:
            records = stages[stage]["records"]
            if not records:
                logger.info("  %s: no data", stage.capitalize())
                continue
            stats = summarize_records(records, args.metrics)
            ratios = summarize_ratios(stages[stage]["ratios"])
            stage_stats[stage] = stats
            ratio_str = ""
            if ratios:
                ratio_str = f" (ratio mean={ratios['mean']:.3f})"
            logger.info("  %s%s", stage.capitalize(), ratio_str)
            if stage == "augmentation" and dataset_entries:
                # Flatten all plot metrics from all groups for summary
                all_plot_metrics = [m for group_metrics in PLOT_METRICS.values() for m in group_metrics]
                ratio_summary = _aggregate_ratio_metrics(dataset_entries, all_plot_metrics)
                if ratio_summary:
                    logger.info("      augmentation ratios:")
                    for ratio_value in sorted(ratio_summary.keys()):
                        metric_summary = ", ".join(
                            f"{metric}={ratio_summary[ratio_value][metric]:.3f}"
                            for metric in all_plot_metrics
                            if metric in ratio_summary[ratio_value] and ratio_summary[ratio_value][metric] is not None
                        )
                        logger.info("        - %0.3f: %s", ratio_value, metric_summary)

            for metric in args.metrics:
                if metric not in stats:
                    continue
                m = stats[metric]
                logger.info(
                    "    - %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, n=%d",
                    metric,
                    m["mean"],
                    m["std"],
                    m["min"],
                    m["max"],
                    m["count"],
                )

        # Advanced statistics if requested
        if args.advanced_stats and "reduction" in stage_stats and "augmentation" in stage_stats:
            logger.info("  Advanced statistics:")
            for metric in args.metrics:
                red_stats = stage_stats.get("reduction", {}).get(metric)
                aug_stats = stage_stats.get("augmentation", {}).get(metric)

                if not red_stats or not aug_stats:
                    continue

                # Collect individual values for statistical tests
                dataset_entries = ratio_entries.get(dataset, [])
                red_values = []
                aug_values = []
                for entry in dataset_entries:
                    red_entry = entry.get("reduction")
                    aug_entry = entry.get("augmentation")
                    if red_entry and metric in red_entry.get("metrics", {}):
                        red_values.append(red_entry["metrics"][metric])
                    if aug_entry and metric in aug_entry.get("metrics", {}):
                        aug_values.append(aug_entry["metrics"][metric])

                # Cohen's d effect size
                if red_values and aug_values:
                    effect_size = cohens_d(red_values, aug_values)
                    if effect_size is not None:
                        logger.info("    - %s: Cohen's d = %.4f", metric, effect_size)

                # Paired t-test
                if len(red_values) == len(aug_values) and len(red_values) >= 2:
                    t_test_result = paired_t_test(red_values, aug_values)
                    if t_test_result:
                        logger.info(
                            "    - %s: t-test t=%.4f, p=%.4f, mean_diff=%.4f",
                            metric,
                            t_test_result["t_statistic"],
                            t_test_result["p_value"],
                            t_test_result["mean_difference"],
                        )

        # Generate comparison bar charts for all metrics with unified colors
        stage_colors_for_plot = {
            "reduction": {"primary": stats_config.stage_color_reduction_primary},
            "augmentation": {"primary": stats_config.stage_color_augmentation_primary},
        }
        for metric in args.metrics:
            plot_dataset(dataset, metric, stage_stats, plots_base, stage_colors_for_plot)

        dataset_stage_stats[dataset] = stage_stats
        aggregated_output[dataset] = {
            stage: {
                "metrics": stage_stats.get(stage, {}),
                "ratios": summarize_ratios(stages[stage]["ratios"]),
            }
            for stage in STAGES
        }
        logger.info("")

    # Export results in requested formats
    export_dir = Path(args.tsv_dir)
    # Ensure export_dir is absolute - resolve relative to PROJECT_ROOT if needed
    if not export_dir.is_absolute():
        export_dir = (PROJECT_ROOT / export_dir).resolve()
    else:
        export_dir = export_dir.resolve()

    # Export results in requested formats
    if "tsv" in args.export_formats:
        write_dataset_summary_tsv(export_dir / "dataset_summary.tsv", dataset_stage_stats, args.metrics)
        write_ratio_summary_tsv(export_dir / "ratio_summary.tsv", ratio_entries, args.metrics)
        logger.info("TSV summaries saved under %s", export_dir.resolve())

    if "latex" in args.export_formats or "latex-doc" in args.export_formats:
        latex_dir = export_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("=" * 80)
        logger.info("Generating LaTeX exports...")
        logger.info("=" * 80)

        # Export LaTeX table
        write_dataset_summary_latex(
            latex_dir / "dataset_summary.tex",
            dataset_stage_stats,
            args.metrics,
            colored=True,
        )
        logger.info("✓ LaTeX summary table: %s", latex_dir / "dataset_summary.tex")

        # Generate comparison tables (one per dataset, aggregated)
        comparison_tables_dir = latex_dir / "comparison_tables"
        num_tables = write_comparison_tables_latex(
            comparison_tables_dir,
            ratio_entries,
            args.metrics,
        )
        if num_tables > 0:
            logger.info("✓ Generated %d aggregated comparison tables:", num_tables)
            logger.info("  📁 %s", comparison_tables_dir)
            logger.info("  Include in LaTeX: \\input{%s/<dataset>.tex}", comparison_tables_dir.relative_to(PROJECT_ROOT))
        else:
            logger.warning("⚠ No comparison tables generated (no ratio data available)")

        # Generate detailed comparison tables (one per dataset per ratio, individual values)
        detailed_tables_dir = latex_dir / "comparison_tables_detailed"
        num_detailed_tables = write_detailed_comparison_tables_latex(
            detailed_tables_dir,
            ratio_entries,
            args.metrics,
        )
        if num_detailed_tables > 0:
            logger.info("✓ Generated %d detailed tables (individual experiment values):", num_detailed_tables)
            logger.info("  📁 %s", detailed_tables_dir)
            logger.info("  Include in LaTeX: \\input{%s/<dataset>_ratio<X.X>_detailed.tex}", detailed_tables_dir.relative_to(PROJECT_ROOT))
        else:
            logger.warning("⚠ No detailed tables generated")

        logger.info("=" * 80)

        # Generate complete LaTeX document if requested
        if "latex-doc" in args.export_formats:
            create_results_document(
                dataset_stage_stats,
                args.metrics,
                plots_base,
                latex_dir / "results_document.tex",
                include_figures=True,
            )
            logger.info("Complete LaTeX document saved to %s", latex_dir / "results_document.tex")

    # Generate plots for each metric group (ranking, classification, etc.)
    for group_name, group_metrics in PLOT_METRICS.items():
        logger.info(f"Generating {group_name} metrics plots...")
        logger.debug(f"[PLOT_GENERATION] Group={group_name}, metrics={group_metrics}")
        logger.debug(f"[PLOT_GENERATION] ratio_entries has {len(ratio_entries)} datasets")
        for ds, entries in ratio_entries.items():
            logger.debug(f"[PLOT_GENERATION]   {ds}: {len(entries)} entries")

        # Get output directory for this group
        group_plots_dir = Path(stats_config.get_plots_output_dir_for_group(group_name))
        # Ensure group_plots_dir is absolute - resolve relative to PROJECT_ROOT if needed
        if not group_plots_dir.is_absolute():
            group_plots_dir = (PROJECT_ROOT / group_plots_dir).resolve()
        else:
            group_plots_dir = group_plots_dir.resolve()
        group_plots_dir.mkdir(parents=True, exist_ok=True)

        # Build and plot ratio data for this group
        ratio_plot_groups = build_ratio_plot_data(ratio_entries, group_metrics)
        for dataset, ratio_group in ratio_plot_groups.items():
            plot_ratio_groups(dataset, ratio_group, group_metrics, group_plots_dir)
            plot_ratio_trends(dataset, ratio_group, group_metrics, group_plots_dir)

        # Build and plot reverse ratio data (reduction varying, augmentation fixed)
        logger.info(f"Generating reduction trend plots for {group_name} metrics...")
        reverse_ratio_plot_groups = build_reverse_ratio_plot_data(ratio_entries, group_metrics)
        for dataset, reverse_ratio_group in reverse_ratio_plot_groups.items():
            plot_reduction_trends(dataset, reverse_ratio_group, group_metrics, group_plots_dir)

    # Generate advanced visualizations if requested
    if args.enable_advanced_plots:
        logger.info("Generating advanced visualizations...")

        # Prepare color configurations from stats_config
        stage_colors = {
            "reduction": {
                "primary": stats_config.stage_color_reduction_primary,
                "light": stats_config.stage_color_reduction_light,
                "linestyle": stats_config.stage_color_reduction_linestyle,
                "marker": stats_config.stage_color_reduction_marker,
                "alpha": stats_config.stage_color_reduction_alpha,
            },
            "augmentation": {
                "primary": stats_config.stage_color_augmentation_primary,
                "light": stats_config.stage_color_augmentation_light,
                "linestyle": stats_config.stage_color_augmentation_linestyle,
                "marker": stats_config.stage_color_augmentation_marker,
                "alpha": stats_config.stage_color_augmentation_alpha,
            },
        }

        delta_colors = {
            "positive": stats_config.delta_color_positive,
            "negative": stats_config.delta_color_negative,
            "neutral": stats_config.delta_color_neutral,
        }

        for dataset, stage_stats in sorted(dataset_stage_stats.items()):
            # Generate radar chart for multi-metric comparison (once per dataset)
            red_metrics = {}
            aug_metrics = {}
            for metric in args.metrics:
                red_stats = stage_stats.get("reduction", {}).get(metric)
                aug_stats = stage_stats.get("augmentation", {}).get(metric)
                if red_stats:
                    red_metrics[metric] = red_stats["mean"]
                if aug_stats:
                    aug_metrics[metric] = aug_stats["mean"]

            if red_metrics or aug_metrics:
                plot_radar_chart(dataset, red_metrics, aug_metrics, args.metrics, plots_base, stage_colors, dpi=args.dpi)

            for metric in args.metrics:
                red_stats = stage_stats.get("reduction", {}).get(metric)
                aug_stats = stage_stats.get("augmentation", {}).get(metric)

                # Collect values for plotting
                red_values = []
                aug_values = []

                # Extract individual values from ratio_entries for more granular plots
                dataset_entries = ratio_entries.get(dataset, [])
                for entry in dataset_entries:
                    red_entry = entry.get("reduction")
                    aug_entry = entry.get("augmentation")
                    if red_entry and metric in red_entry.get("metrics", {}):
                        red_values.append(red_entry["metrics"][metric])
                    if aug_entry and metric in aug_entry.get("metrics", {}):
                        aug_values.append(aug_entry["metrics"][metric])

                # Generate ridge plot for distribution comparison
                if red_values or aug_values:
                    plot_ridge(dataset, red_values, aug_values, metric, plots_base, stage_colors, dpi=args.dpi)

                # Generate boxplot
                if red_values or aug_values:
                    plot_boxplot(dataset, red_values, aug_values, metric, plots_base, stage_colors, dpi=args.dpi)

                # Generate violin plot
                if red_values or aug_values:
                    plot_violin(dataset, red_values, aug_values, metric, plots_base, stage_colors, dpi=args.dpi)

                # Generate scatter plot (requires paired values)
                if len(red_values) == len(aug_values) and len(red_values) >= 2:
                    plot_scatter_correlation(dataset, red_values, aug_values, metric, plots_base, dpi=args.dpi)

                # Generate delta chart
                if len(red_values) == len(aug_values) and red_values:
                    plot_delta_chart(dataset, red_values, aug_values, metric, plots_base, delta_colors, dpi=args.dpi)

        # Generate heatmaps for ratio combinations
        for dataset, ratio_group in ratio_plot_groups.items():
            for metric in args.metrics:
                heatmap_data = {}
                for red_ratio, aug_dict in ratio_group.items():
                    heatmap_data[red_ratio] = {}
                    for aug_ratio, stage_data in aug_dict.items():
                        aug_value = stage_data.get("augmentation", {}).get(metric)
                        if aug_value is not None:
                            heatmap_data[red_ratio][aug_ratio] = aug_value

                if heatmap_data:
                    plot_heatmap(dataset, heatmap_data, metric, plots_base, dpi=args.dpi)

        # Generate 3D surface plots for both stages
        logger.info("Generating 3D surface plots...")
        for dataset, ratio_group in ratio_plot_groups.items():
            for metric in args.metrics:
                # Surface plot for reduction stage
                plot_surface_3d(
                    dataset,
                    ratio_group,
                    metric,
                    "reduction",
                    plots_base,
                    stage_colors,
                    dpi=args.dpi,
                )
                # Surface plot for augmentation stage
                plot_surface_3d(
                    dataset,
                    ratio_group,
                    metric,
                    "augmentation",
                    plots_base,
                    stage_colors,
                    dpi=args.dpi,
                )
                # Try interactive plots if plotly is available
                plot_interactive_surface_3d(
                    dataset,
                    ratio_group,
                    metric,
                    "augmentation",
                    plots_base,
                    dpi=args.dpi,
                )

        # Generate performance matrices (dataset x experiment heatmaps)
        for metric in args.metrics:
            # Build reduction stage data
            red_stage_data = {}
            for dataset, entries in ratio_entries.items():
                for idx, entry in enumerate(entries):
                    red_entry = entry.get("reduction")
                    if red_entry and metric in red_entry.get("metrics", {}):
                        if dataset not in red_stage_data:
                            red_stage_data[dataset] = {}
                        # Use experiment index or ratio as identifier
                        exp_id = entry.get("experiment", f"exp_{idx}")
                        red_stage_data[dataset][exp_id] = red_entry["metrics"][metric]

            if red_stage_data:
                plot_performance_matrix(red_stage_data, metric, "reduction", plots_base, stage_colors, dpi=args.dpi)

            # Build augmentation stage data
            aug_stage_data = {}
            for dataset, entries in ratio_entries.items():
                for idx, entry in enumerate(entries):
                    aug_entry = entry.get("augmentation")
                    if aug_entry and metric in aug_entry.get("metrics", {}):
                        if dataset not in aug_stage_data:
                            aug_stage_data[dataset] = {}
                        # Use experiment index or ratio as identifier
                        exp_id = entry.get("experiment", f"exp_{idx}")
                        aug_stage_data[dataset][exp_id] = aug_entry["metrics"][metric]

            if aug_stage_data:
                plot_performance_matrix(aug_stage_data, metric, "augmentation", plots_base, stage_colors, dpi=args.dpi)

        logger.info("Advanced visualizations saved under %s/{heatmaps,boxplots,violins,scatter,deltas,radar_charts,ridge_plots,performance_matrices}", plots_base.resolve())

    if args.output_json:
        out_path = Path(args.output_json)
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(aggregated_output, indent=2), encoding="utf-8")
        logger.info("Aggregated data saved to %s", out_path.resolve())

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Analysis complete!")
    logger.info("=" * 80)
    logger.info("Output directory: %s", export_dir.resolve())
    logger.info("")
    logger.info("Generated files:")

    if "tsv" in args.export_formats:
        logger.info("  📄 TSV exports:")
        logger.info("     - dataset_summary.tsv")
        logger.info("     - ratio_summary.tsv")

    if "latex" in args.export_formats or "latex-doc" in args.export_formats:
        logger.info("  📄 LaTeX exports:")
        logger.info("     - latex/dataset_summary.tex")
        if ratio_entries:
            logger.info("     - latex/comparison_tables/ (%d aggregated tables)", len(ratio_entries))
            # Count detailed tables
            num_detailed = sum(len(ratios) for ratios in
                             [{ratio for entry in entries if (red := entry.get("reduction")) and red.get("ratio") for ratio in [round(float(red.get("ratio")), 6)]}
                              for dataset, entries in ratio_entries.items()])
            if num_detailed > 0:
                logger.info("     - latex/comparison_tables_detailed/ (individual experiment values)")
        if "latex-doc" in args.export_formats:
            logger.info("     - latex/results_document.tex")

    logger.info("  📊 Plots: %s", Path(args.plots_dir).resolve())
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
