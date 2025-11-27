#!/usr/bin/env python3
"""Advanced visualization utilities - heatmaps, boxplots, violin plots, scatter plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_heatmap(
    dataset: str,
    data: Dict[float, Dict[float, float]],
    metric: str,
    plots_dir: Path,
    dpi: int = 200,
) -> None:
    """Create heatmap showing metric values across reduction/augmentation ratios."""
    if not data:
        return

    reduction_ratios = sorted(data.keys())
    augmentation_ratios_set = set()
    for aug_dict in data.values():
        augmentation_ratios_set.update(aug_dict.keys())
    augmentation_ratios = sorted(augmentation_ratios_set)

    if not reduction_ratios or not augmentation_ratios:
        return

    # Create matrix
    matrix = np.zeros((len(reduction_ratios), len(augmentation_ratios)))
    for i, red_ratio in enumerate(reduction_ratios):
        for j, aug_ratio in enumerate(augmentation_ratios):
            value = data.get(red_ratio, {}).get(aug_ratio)
            matrix[i, j] = value if value is not None else np.nan

    ensure_dir(plots_dir / "heatmaps")

    fig, ax = plt.subplots(figsize=(max(8, len(augmentation_ratios)), max(6, len(reduction_ratios))))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", interpolation="nearest")

    # Set ticks
    ax.set_xticks(np.arange(len(augmentation_ratios)))
    ax.set_yticks(np.arange(len(reduction_ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in augmentation_ratios])
    ax.set_yticklabels([f"{r:.2f}" for r in reduction_ratios])

    # Labels
    ax.set_xlabel("Augmentation Ratio", fontsize=11)
    ax.set_ylabel("Reduction Ratio", fontsize=11)
    ax.set_title(f"{dataset} - {metric} Heatmap", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(reduction_ratios)):
        for j in range(len(augmentation_ratios)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=15)

    plt.tight_layout()
    outfile = plots_dir / "heatmaps" / f"{dataset}_{metric.replace('@', 'at')}_heatmap.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_boxplot(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]] = None,
    dpi: int = 200,
) -> None:
    """Create boxplot comparing reduction vs augmentation."""
    if not reduction_values and not augmentation_values:
        return

    # Default colors if not provided
    if stage_colors is None:
        stage_colors = {
            "reduction": {"primary": "#264653"},
            "augmentation": {"primary": "#e76f51"},
        }

    ensure_dir(plots_dir / "boxplots")

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    labels = []
    colors = []

    if reduction_values:
        data.append(reduction_values)
        labels.append("Reduction")
        colors.append(stage_colors.get("reduction", {}).get("primary", "#264653"))

    if augmentation_values:
        data.append(augmentation_values)
        labels.append("Augmentation")
        colors.append(stage_colors.get("augmentation", {}).get("primary", "#e76f51"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True,
                     showmeans=True, meanline=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{dataset} - {metric} Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "boxplots" / f"{dataset}_{metric.replace('@', 'at')}_boxplot.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_violin(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]] = None,
    dpi: int = 200,
) -> None:
    """Create violin plot comparing reduction vs augmentation."""
    if (not reduction_values or len(reduction_values) < 2) and (not augmentation_values or len(augmentation_values) < 2):
        return

    # Default colors if not provided
    if stage_colors is None:
        stage_colors = {
            "reduction": {"primary": "#264653"},
            "augmentation": {"primary": "#e76f51"},
        }

    ensure_dir(plots_dir / "violins")

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    positions = []
    labels = []
    colors = []

    pos = 1
    if reduction_values and len(reduction_values) >= 2:
        data.append(reduction_values)
        positions.append(pos)
        labels.append("Reduction")
        colors.append(stage_colors.get("reduction", {}).get("primary", "#264653"))
        pos += 1

    if augmentation_values and len(augmentation_values) >= 2:
        data.append(augmentation_values)
        positions.append(pos)
        labels.append("Augmentation")
        colors.append(stage_colors.get("augmentation", {}).get("primary", "#e76f51"))

    if not data:
        return

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        if i < len(colors):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{dataset} - {metric} Violin Plot", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "violins" / f"{dataset}_{metric.replace('@', 'at')}_violin.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_scatter_correlation(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    dpi: int = 200,
) -> None:
    """Create scatter plot showing correlation between reduction and augmentation."""
    if len(reduction_values) != len(augmentation_values) or len(reduction_values) < 2:
        return

    ensure_dir(plots_dir / "scatter")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(reduction_values, augmentation_values, alpha=0.6, s=80, c="#457b9d", edgecolors="black", linewidths=0.5)

    # Add diagonal reference line (y=x)
    min_val = min(min(reduction_values), min(augmentation_values))
    max_val = max(max(reduction_values), max(augmentation_values))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="y=x (no change)")

    # Calculate and display correlation
    if len(reduction_values) > 1:
        corr = np.corrcoef(reduction_values, augmentation_values)[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel(f"Reduction {metric}", fontsize=11)
    ax.set_ylabel(f"Augmentation {metric}", fontsize=11)
    ax.set_title(f"{dataset} - {metric} Correlation", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "scatter" / f"{dataset}_{metric.replace('@', 'at')}_scatter.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_delta_chart(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    delta_colors: Dict[str, str] = None,
    dpi: int = 200,
) -> None:
    """Create chart showing delta (augmentation - reduction) for each experiment."""
    if len(reduction_values) != len(augmentation_values) or not reduction_values:
        return

    # Default delta colors if not provided
    if delta_colors is None:
        delta_colors = {
            "positive": "#2a9d8f",
            "negative": "#e63946",
            "neutral": "#6c757d",
        }

    ensure_dir(plots_dir / "deltas")

    deltas = [aug - red for red, aug in zip(reduction_values, augmentation_values)]
    indices = list(range(len(deltas)))

    fig, ax = plt.subplots(figsize=(max(8, len(deltas) * 0.5), 5))

    colors = [delta_colors.get("positive", "#2a9d8f") if d > 0 else delta_colors.get("negative", "#e63946") for d in deltas]
    ax.bar(indices, deltas, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add mean line
    mean_delta = np.mean(deltas)
    ax.axhline(y=mean_delta, color="blue", linestyle="--", linewidth=1, label=f"Mean Δ: {mean_delta:.4f}")

    ax.set_xlabel("Experiment Index", fontsize=11)
    ax.set_ylabel(f"Δ {metric} (Augmentation - Reduction)", fontsize=11)
    ax.set_title(f"{dataset} - {metric} Performance Delta", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outfile = plots_dir / "deltas" / f"{dataset}_{metric.replace('@', 'at')}_delta.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_radar_chart(
    dataset: str,
    reduction_metrics: Dict[str, float],
    augmentation_metrics: Dict[str, float],
    metrics: List[str],
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create radar/spider chart comparing reduction vs augmentation metrics.

    Args:
        dataset: Dataset name
        reduction_metrics: Dictionary of metric -> value for reduction
        augmentation_metrics: Dictionary of metric -> value for augmentation
        metrics: List of metrics to plot
        plots_dir: Output directory
        stage_colors: Color configuration for reduction/augmentation
        dpi: Resolution
    """
    # Filter to metrics that have values
    valid_metrics = [m for m in metrics if m in reduction_metrics or m in augmentation_metrics]
    if not valid_metrics:
        return

    ensure_dir(plots_dir / "radar_charts")

    num_vars = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Prepare data
    red_values = [reduction_metrics.get(m, 0) for m in valid_metrics]
    aug_values = [augmentation_metrics.get(m, 0) for m in valid_metrics]
    red_values += red_values[:1]  # Complete the circle
    aug_values += aug_values[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))

    # Plot reduction
    red_color = stage_colors.get("reduction", {}).get("primary", "#264653")
    aug_color = stage_colors.get("augmentation", {}).get("primary", "#e76f51")

    ax.plot(angles, red_values, 'o-', linewidth=2, label='Reduction',
            color=red_color, alpha=0.7)
    ax.fill(angles, red_values, alpha=0.15, color=red_color)

    # Plot augmentation
    ax.plot(angles, aug_values, 'o-', linewidth=2, label='Augmentation',
            color=aug_color, alpha=0.7)
    ax.fill(angles, aug_values, alpha=0.15, color=aug_color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{dataset} - Multi-Metric Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    outfile = plots_dir / "radar_charts" / f"{dataset}_radar.png"
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_ridge(
    dataset: str,
    reduction_values: List[float],
    augmentation_values: List[float],
    metric: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create ridge plot showing distribution comparison.

    Args:
        dataset: Dataset name
        reduction_values: Values for reduction
        augmentation_values: Values for augmentation
        metric: Metric name
        plots_dir: Output directory
        stage_colors: Color configuration
        dpi: Resolution
    """
    if (not reduction_values or len(reduction_values) < 2) and \
       (not augmentation_values or len(augmentation_values) < 2):
        return

    ensure_dir(plots_dir / "ridge_plots")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    red_color = stage_colors.get("reduction", {}).get("primary", "#264653")
    aug_color = stage_colors.get("augmentation", {}).get("primary", "#e76f51")

    # Reduction distribution
    if reduction_values and len(reduction_values) >= 2:
        axes[0].fill_between(
            np.linspace(min(reduction_values), max(reduction_values), 100),
            0,
            np.histogram(reduction_values, bins=30, density=True)[0].max(),
            alpha=0.6,
            color=red_color,
            label='Reduction'
        )
        axes[0].hist(reduction_values, bins=30, density=True, alpha=0.7,
                     color=red_color, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel("Density", fontsize=10)
        axes[0].legend(loc='upper right')
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Augmentation distribution
    if augmentation_values and len(augmentation_values) >= 2:
        axes[1].fill_between(
            np.linspace(min(augmentation_values), max(augmentation_values), 100),
            0,
            np.histogram(augmentation_values, bins=30, density=True)[0].max(),
            alpha=0.6,
            color=aug_color,
            label='Augmentation'
        )
        axes[1].hist(augmentation_values, bins=30, density=True, alpha=0.7,
                     color=aug_color, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel(metric, fontsize=11)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].legend(loc='upper right')
        axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    fig.suptitle(f"{dataset} - {metric} Distribution Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    outfile = plots_dir / "ridge_plots" / f"{dataset}_{metric.replace('@', 'at')}_ridge.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_performance_matrix(
    stage_data: Dict[str, Dict[str, float]],
    metric: str,
    stage: str,
    plots_dir: Path,
    stage_colors: Dict[str, Dict[str, str]],
    dpi: int = 200,
) -> None:
    """Create performance matrix heatmap (dataset × model).

    Args:
        stage_data: Dict of dataset -> dict of model -> metric value
        metric: Metric name
        stage: Stage name ("reduction" or "augmentation")
        plots_dir: Output directory
        stage_colors: Color configuration
        dpi: Resolution
    """
    if not stage_data:
        return

    ensure_dir(plots_dir / "performance_matrices")

    # Prepare data
    datasets = sorted(stage_data.keys())
    all_models = set()
    for models_dict in stage_data.values():
        all_models.update(models_dict.keys())
    models = sorted(all_models)

    if not datasets or not models:
        return

    # Create matrix
    matrix = np.zeros((len(datasets), len(models)))
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            value = stage_data.get(dataset, {}).get(model)
            matrix[i, j] = value if value is not None else np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), max(8, len(datasets) * 0.5)))

    # Use appropriate colormap based on stage
    stage_color = stage_colors.get(stage, {}).get("primary", "#264653")
    cmap = "YlOrRd" if stage == "augmentation" else "YlGnBu"

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(datasets)

    # Labels
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_title(f"{stage.capitalize()} - {metric} Performance Matrix",
                 fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(models)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.3f}",
                       ha="center", va="center", color=text_color, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=20)

    plt.tight_layout()
    outfile = plots_dir / "performance_matrices" / f"{stage}_{metric.replace('@', 'at')}_matrix.png"
    plt.savefig(outfile, dpi=dpi)
    plt.close(fig)


__all__ = [
    "plot_heatmap",
    "plot_boxplot",
    "plot_violin",
    "plot_scatter_correlation",
    "plot_delta_chart",
    "plot_radar_chart",
    "plot_ridge",
    "plot_performance_matrix",
]
