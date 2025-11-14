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
    dpi: int = 200,
) -> None:
    """Create boxplot comparing reduction vs augmentation."""
    if not reduction_values and not augmentation_values:
        return

    ensure_dir(plots_dir / "boxplots")

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    labels = []
    colors = []

    if reduction_values:
        data.append(reduction_values)
        labels.append("Reduction")
        colors.append("#457b9d")

    if augmentation_values:
        data.append(augmentation_values)
        labels.append("Augmentation")
        colors.append("#e76f51")

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
    dpi: int = 200,
) -> None:
    """Create violin plot comparing reduction vs augmentation."""
    if (not reduction_values or len(reduction_values) < 2) and (not augmentation_values or len(augmentation_values) < 2):
        return

    ensure_dir(plots_dir / "violins")

    fig, ax = plt.subplots(figsize=(6, 5))

    data = []
    positions = []
    labels = []

    pos = 1
    if reduction_values and len(reduction_values) >= 2:
        data.append(reduction_values)
        positions.append(pos)
        labels.append("Reduction")
        pos += 1

    if augmentation_values and len(augmentation_values) >= 2:
        data.append(augmentation_values)
        positions.append(pos)
        labels.append("Augmentation")

    if not data:
        return

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    colors = ["#457b9d", "#e76f51"]
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
    dpi: int = 200,
) -> None:
    """Create chart showing delta (augmentation - reduction) for each experiment."""
    if len(reduction_values) != len(augmentation_values) or not reduction_values:
        return

    ensure_dir(plots_dir / "deltas")

    deltas = [aug - red for red, aug in zip(reduction_values, augmentation_values)]
    indices = list(range(len(deltas)))

    fig, ax = plt.subplots(figsize=(max(8, len(deltas) * 0.5), 5))

    colors = ["#2a9d8f" if d > 0 else "#e76f51" for d in deltas]
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


__all__ = [
    "plot_heatmap",
    "plot_boxplot",
    "plot_violin",
    "plot_scatter_correlation",
    "plot_delta_chart",
]
