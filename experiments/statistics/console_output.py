#!/usr/bin/env python3
"""Rich console output utilities for statistics analysis."""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Progress = None


def create_console(use_rich: bool = True) -> Optional[Any]:
    """Create rich console if available.

    Args:
        use_rich: Whether to use rich output

    Returns:
        Rich Console instance or None
    """
    if use_rich and RICH_AVAILABLE:
        return Console()
    return None


def create_progress_bar() -> Optional[Any]:
    """Create rich progress bar if available.

    Returns:
        Rich Progress instance or None
    """
    if not RICH_AVAILABLE:
        return None

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=Console(),
    )


def print_header(
    console: Optional[Any],
    text: str,
    style: str = "cyan bold",
) -> None:
    """Print formatted header.

    Args:
        console: Rich console instance or None
        text: Header text
        style: Rich style string
    """
    if console and RICH_AVAILABLE:
        console.print(f"\n[{style}]{text}[/{style}]")
    else:
        print(f"\n=== {text} ===")


def print_dataset_header(
    console: Optional[Any],
    dataset: str,
    style: str = "blue bold",
) -> None:
    """Print formatted dataset header.

    Args:
        console: Rich console instance or None
        dataset: Dataset name
        style: Rich style string
    """
    if console and RICH_AVAILABLE:
        console.print(f"\n[{style}]{'='*80}[/{style}]")
        console.print(f"[{style}]Dataset: {dataset}[/{style}]")
        console.print(f"[{style}]{'='*80}[/{style}]")
    else:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")


def format_delta_with_color(
    value: float,
    thresholds: Dict[str, float],
    colors: Dict[str, str],
) -> str:
    """Format delta value with color based on thresholds.

    Args:
        value: Delta value
        thresholds: Dict with 'significant_improvement' and 'significant_degradation' keys
        colors: Dict with 'improvement', 'degradation', 'neutral' keys

    Returns:
        Formatted string with rich color tags (if rich available) or plain text
    """
    if not RICH_AVAILABLE:
        return f"{value:+.4f}"

    sig_improvement = thresholds.get("significant_improvement", 0.05)
    sig_degradation = thresholds.get("significant_degradation", -0.05)

    if value >= sig_improvement:
        color = colors.get("improvement", "green")
        symbol = "↑"
        return f"[{color}]{symbol} {value:+.4f}[/{color}]"
    elif value <= sig_degradation:
        color = colors.get("degradation", "red")
        symbol = "↓"
        return f"[{color}]{symbol} {value:+.4f}[/{color}]"
    else:
        color = colors.get("neutral", "yellow")
        return f"[{color}]{value:+.4f}[/{color}]"


def print_metric_summary(
    console: Optional[Any],
    stage: str,
    metric: str,
    stats: Dict[str, float],
    ratio_str: str = "",
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Print metric summary with formatting.

    Args:
        console: Rich console instance or None
        stage: Stage name ("reduction" or "augmentation")
        metric: Metric name
        stats: Stats dict with mean, std, min, max, count
        ratio_str: Optional ratio information string
        colors: Color configuration
    """
    if colors is None:
        colors = {}

    metric_color = colors.get("metric", "white")
    value_color = colors.get("value", "white")

    if console and RICH_AVAILABLE:
        console.print(
            f"    [{metric_color}]- {metric}[/{metric_color}]: "
            f"[{value_color}]mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, "
            f"max={stats['max']:.4f}, "
            f"n={stats['count']}[/{value_color}]"
        )
    else:
        print(
            f"    - {metric}: "
            f"mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, "
            f"max={stats['max']:.4f}, "
            f"n={stats['count']}"
        )


def create_summary_table(
    console: Optional[Any],
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    thresholds: Dict[str, float],
    colors: Dict[str, str],
    top_n: int = 5,
    sort_by: str = "hits@1",
) -> None:
    """Create and print summary table with top/bottom experiments.

    Args:
        console: Rich console instance or None
        dataset_stage_stats: Dataset statistics
        metrics: List of metrics to display
        thresholds: Delta thresholds for coloring
        colors: Color configuration
        top_n: Number of top/bottom datasets to show
        sort_by: Metric to sort by
    """
    if not console or not RICH_AVAILABLE:
        print_plain_summary_table(dataset_stage_stats, metrics, top_n, sort_by)
        return

    # Calculate deltas for sorting
    dataset_deltas = []
    for dataset, stage_stats in dataset_stage_stats.items():
        red_stats = stage_stats.get("reduction", {}).get(sort_by)
        aug_stats = stage_stats.get("augmentation", {}).get(sort_by)

        if red_stats and aug_stats:
            delta = aug_stats["mean"] - red_stats["mean"]
            delta_pct = (delta / red_stats["mean"] * 100) if red_stats["mean"] != 0 else 0
            dataset_deltas.append({
                "dataset": dataset,
                "red_mean": red_stats["mean"],
                "aug_mean": aug_stats["mean"],
                "delta": delta,
                "delta_pct": delta_pct,
            })

    if not dataset_deltas:
        return

    # Sort by delta
    dataset_deltas.sort(key=lambda x: x["delta"], reverse=True)

    # Top N improvements
    console.print(f"\n[cyan bold]{'='*80}[/cyan bold]")
    console.print(f"[cyan bold]Summary: Top {top_n} Improvements (by {sort_by})[/cyan bold]")
    console.print(f"[cyan bold]{'='*80}[/cyan bold]")

    table_top = Table(show_header=True, header_style="bold cyan")
    table_top.add_column("Rank", justify="right", style="cyan")
    table_top.add_column("Dataset", style="blue bold")
    table_top.add_column(f"Red {sort_by}", justify="right")
    table_top.add_column(f"Aug {sort_by}", justify="right")
    table_top.add_column("Δ", justify="right")
    table_top.add_column("Δ%", justify="right")

    for i, entry in enumerate(dataset_deltas[:top_n], 1):
        delta_str = format_delta_with_color(entry["delta"], thresholds, colors)
        delta_pct_str = format_delta_with_color(entry["delta_pct"] / 100, thresholds, colors)

        table_top.add_row(
            str(i),
            entry["dataset"],
            f"{entry['red_mean']:.4f}",
            f"{entry['aug_mean']:.4f}",
            delta_str,
            f"{entry['delta_pct']:+.2f}%",
        )

    console.print(table_top)

    # Bottom N (worst/degradations)
    if len(dataset_deltas) > top_n:
        console.print(f"\n[cyan bold]Top {top_n} Degradations (by {sort_by})[/cyan bold]")

        table_bottom = Table(show_header=True, header_style="bold cyan")
        table_bottom.add_column("Rank", justify="right", style="cyan")
        table_bottom.add_column("Dataset", style="blue bold")
        table_bottom.add_column(f"Red {sort_by}", justify="right")
        table_bottom.add_column(f"Aug {sort_by}", justify="right")
        table_bottom.add_column("Δ", justify="right")
        table_bottom.add_column("Δ%", justify="right")

        for i, entry in enumerate(reversed(dataset_deltas[-top_n:]), 1):
            delta_str = format_delta_with_color(entry["delta"], thresholds, colors)
            delta_pct_str = format_delta_with_color(entry["delta_pct"] / 100, thresholds, colors)

            table_bottom.add_row(
                str(i),
                entry["dataset"],
                f"{entry['red_mean']:.4f}",
                f"{entry['aug_mean']:.4f}",
                delta_str,
                f"{entry['delta_pct']:+.2f}%",
            )

        console.print(table_bottom)


def print_plain_summary_table(
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    top_n: int,
    sort_by: str,
) -> None:
    """Print plain text summary table (fallback when rich not available)."""
    # Calculate deltas
    dataset_deltas = []
    for dataset, stage_stats in dataset_stage_stats.items():
        red_stats = stage_stats.get("reduction", {}).get(sort_by)
        aug_stats = stage_stats.get("augmentation", {}).get(sort_by)

        if red_stats and aug_stats:
            delta = aug_stats["mean"] - red_stats["mean"]
            delta_pct = (delta / red_stats["mean"] * 100) if red_stats["mean"] != 0 else 0
            dataset_deltas.append({
                "dataset": dataset,
                "red_mean": red_stats["mean"],
                "aug_mean": aug_stats["mean"],
                "delta": delta,
                "delta_pct": delta_pct,
            })

    if not dataset_deltas:
        return

    dataset_deltas.sort(key=lambda x: x["delta"], reverse=True)

    print(f"\n{'='*80}")
    print(f"Summary: Top {top_n} Improvements (by {sort_by})")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Dataset':<30} {'Red':<10} {'Aug':<10} {'Delta':<10} {'Delta%':<10}")
    print("-" * 80)

    for i, entry in enumerate(dataset_deltas[:top_n], 1):
        print(
            f"{i:<6} {entry['dataset']:<30} "
            f"{entry['red_mean']:<10.4f} {entry['aug_mean']:<10.4f} "
            f"{entry['delta']:<+10.4f} {entry['delta_pct']:<+10.2f}%"
        )

    if len(dataset_deltas) > top_n:
        print(f"\nTop {top_n} Degradations (by {sort_by})")
        print("-" * 80)
        for i, entry in enumerate(reversed(dataset_deltas[-top_n:]), 1):
            print(
                f"{i:<6} {entry['dataset']:<30} "
                f"{entry['red_mean']:<10.4f} {entry['aug_mean']:<10.4f} "
                f"{entry['delta']:<+10.4f} {entry['delta_pct']:<+10.2f}%"
            )


__all__ = [
    "create_console",
    "create_progress_bar",
    "print_header",
    "print_dataset_header",
    "format_delta_with_color",
    "print_metric_summary",
    "create_summary_table",
    "RICH_AVAILABLE",
]
