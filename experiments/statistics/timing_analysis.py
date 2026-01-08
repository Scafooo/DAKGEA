#!/usr/bin/env python3
"""Timing analysis module for experiment runs.

This module analyzes folder metadata to determine execution time for each
stage of the experiment pipeline (reduction, augmentation, model training).

The timing is computed from folder birth (creation) and modification timestamps:
- Start time: Folder birth time (when the folder was first created)
- End time: Folder modification time (when the last file was written)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from experiments.statistics.config import get_statistics_config


@dataclass
class StageTiming:
    """Timing information for a single stage."""

    stage: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        if self.start_time and self.end_time and self.duration_seconds is None:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    @property
    def duration(self) -> Optional[timedelta]:
        """Get duration as timedelta."""
        if self.duration_seconds is not None:
            return timedelta(seconds=self.duration_seconds)
        return None

    @property
    def duration_formatted(self) -> str:
        """Get human-readable duration string."""
        if self.duration_seconds is None:
            return "N/A"

        total_seconds = int(self.duration_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self.duration_formatted,
        }


@dataclass
class ExperimentTiming:
    """Timing information for an entire experiment run."""

    experiment_path: Path
    experiment_name: str
    stages: Dict[str, StageTiming] = field(default_factory=dict)
    total_duration_seconds: Optional[float] = None

    @property
    def total_duration(self) -> Optional[timedelta]:
        """Get total duration as timedelta."""
        if self.total_duration_seconds is not None:
            return timedelta(seconds=self.total_duration_seconds)
        return None

    @property
    def total_duration_formatted(self) -> str:
        """Get human-readable total duration string."""
        if self.total_duration_seconds is None:
            return "N/A"

        total_seconds = int(self.total_duration_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_path": str(self.experiment_path),
            "experiment_name": self.experiment_name,
            "stages": {name: timing.to_dict() for name, timing in self.stages.items()},
            "total_duration_seconds": self.total_duration_seconds,
            "total_duration_formatted": self.total_duration_formatted,
        }


def get_folder_timestamps(folder_path: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get start and end timestamps for a stage folder.

    Uses results.json modification time as end time (more reliable than folder mtime).
    Uses folder birth time or earliest file as start time.

    Args:
        folder_path: Path to the stage folder

    Returns:
        Tuple of (start_time, end_time)
    """
    if not folder_path.exists():
        return None, None

    # End time: Use results.json modification time (most reliable)
    results_file = folder_path / "results.json"
    end_time = None
    if results_file.exists():
        results_stat = results_file.stat()
        end_time = datetime.fromtimestamp(results_stat.st_mtime)
    else:
        # Fallback: use folder modification time if results.json doesn't exist
        stat_result = folder_path.stat()
        end_time = datetime.fromtimestamp(stat_result.st_mtime)

    # Start time: Try to get folder birth time (creation time)
    start_time = None
    stat_result = folder_path.stat()

    # Try st_birthtime first (macOS)
    if hasattr(stat_result, 'st_birthtime'):
        start_time = datetime.fromtimestamp(stat_result.st_birthtime)
    else:
        # On Linux, use the stat command to get birth time
        import subprocess
        try:
            result = subprocess.run(
                ['stat', '-c', '%W', str(folder_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                btime_str = result.stdout.strip()
                # %W returns epoch seconds, or '-' or '0' if not available
                if btime_str and btime_str not in ('-', '0'):
                    btime_float = float(btime_str)
                    if btime_float > 0:
                        start_time = datetime.fromtimestamp(btime_float)
        except Exception:
            pass

    # If we couldn't get birth time, find earliest file in folder
    if start_time is None:
        earliest_time = end_time
        try:
            for item in folder_path.iterdir():
                if item.is_file():  # Only check files
                    item_stat = item.stat()
                    item_mtime = datetime.fromtimestamp(item_stat.st_mtime)
                    if earliest_time is None or item_mtime < earliest_time:
                        earliest_time = item_mtime
        except Exception:
            pass
        start_time = earliest_time

    # Safety check: ensure start_time <= end_time
    # If start > end, it means we have copied data (birth time is newer than modification time)
    # In this case, use earliest file modification time as start_time
    if start_time and end_time and start_time > end_time:
        earliest_mtime = None
        try:
            for item in folder_path.iterdir():
                if item.is_file():
                    item_stat = item.stat()
                    item_mtime = datetime.fromtimestamp(item_stat.st_mtime)
                    if earliest_mtime is None or item_mtime < earliest_mtime:
                        earliest_mtime = item_mtime
        except Exception:
            pass

        # Use earliest file mtime if found and it's before end_time
        if earliest_mtime and earliest_mtime <= end_time:
            start_time = earliest_mtime
        else:
            # Last resort: set start = end (0 duration)
            start_time = end_time

    return start_time, end_time


def analyze_experiment_timing(experiment_path: Path) -> ExperimentTiming:
    """Analyze timing for a single experiment.

    Args:
        experiment_path: Path to the experiment directory

    Returns:
        ExperimentTiming object with timing information
    """
    config = get_statistics_config()
    stages = config.stages

    timing = ExperimentTiming(
        experiment_path=experiment_path,
        experiment_name=experiment_path.name,
    )

    earliest_start = None
    latest_end = None
    previous_stage_end = None

    # First pass: collect all stage timings without sequential inference
    for stage in stages:
        stage_path = experiment_path / stage
        if not stage_path.exists():
            continue

        birth_time, mod_time = get_folder_timestamps(stage_path)

        stage_timing = StageTiming(
            stage=stage,
            start_time=birth_time,
            end_time=mod_time,
        )
        timing.stages[stage] = stage_timing

        # Track overall experiment timing
        if birth_time:
            if earliest_start is None or birth_time < earliest_start:
                earliest_start = birth_time
        if mod_time:
            if latest_end is None or mod_time > latest_end:
                latest_end = mod_time

    # Calculate total duration
    if earliest_start and latest_end:
        timing.total_duration_seconds = (latest_end - earliest_start).total_seconds()

    # Handle copied data where all stages have 0 duration (start_time == end_time)
    # Use end time differences to infer stage durations
    zero_duration_stages = [
        (name, st) for name, st in timing.stages.items()
        if st.duration_seconds == 0 and st.start_time and st.end_time and st.start_time == st.end_time
    ]

    if len(zero_duration_stages) > 0 and timing.total_duration_seconds and timing.total_duration_seconds > 1:
        # Sort stages by end time
        sorted_stages = sorted(zero_duration_stages, key=lambda x: x[1].end_time)

        # Infer durations based on time between stage completions
        for i in range(len(sorted_stages)):
            stage_name, stage_timing = sorted_stages[i]

            if i < len(sorted_stages) - 1:
                # Not the last stage: duration = time to next stage's end
                next_stage_timing = sorted_stages[i + 1][1]
                duration_seconds = (next_stage_timing.end_time - stage_timing.end_time).total_seconds()
                # Adjust start time to be just after previous stage (or at earliest_start for first stage)
                if i == 0:
                    stage_timing.start_time = earliest_start
                else:
                    prev_stage = sorted_stages[i - 1][1]
                    stage_timing.start_time = prev_stage.end_time
                # Keep end time as-is
                stage_timing.duration_seconds = (stage_timing.end_time - stage_timing.start_time).total_seconds()
            else:
                # Last stage: gets remaining time to latest_end
                if i == 0:
                    # Only one stage: gets full duration
                    stage_timing.start_time = earliest_start
                else:
                    # Use previous stage's end as start
                    prev_stage = sorted_stages[i - 1][1]
                    stage_timing.start_time = prev_stage.end_time
                stage_timing.end_time = latest_end
                stage_timing.duration_seconds = (stage_timing.end_time - stage_timing.start_time).total_seconds()

    return timing


def apply_sequential_timing_inference(timings: Dict[str, ExperimentTiming]) -> None:
    """Apply sequential timing inference for experiments run in sequence.

    For experiments of the same dataset/ratio with consecutive seeds,
    use the end time of experiment N-1 as the start time of experiment N.
    This helps recover timing information for copied data.

    Args:
        timings: Dictionary of experiment timings (modified in place)
    """
    from collections import defaultdict
    import re

    # Group experiments by dataset and ratio
    # Expected format: DATASET_RATIO_SEED (e.g., BBC_DB_05_03, ICEW_YAGO_04_05)
    groups = defaultdict(list)

    for exp_name, exp_timing in timings.items():
        # Extract dataset, ratio, and seed from experiment name
        # Match patterns like: BBC_DB_05_03, D_W_15K_V1_04_05, ICEW_YAGO_04_05
        match = re.match(r'^(.+?)_(\d{2})_(\d{2})$', exp_name)
        if match:
            dataset = match.group(1)
            ratio = match.group(2)
            seed = match.group(3)
            group_key = f"{dataset}_{ratio}"
            groups[group_key].append((int(seed), exp_name, exp_timing))

    config = get_statistics_config()
    stages = config.stages

    # Process each group
    for group_key, experiments in groups.items():
        # Sort by seed
        experiments.sort(key=lambda x: x[0])

        # Apply sequential inference
        for i in range(len(experiments)):
            seed, exp_name, exp_timing = experiments[i]

            # Check if this experiment has zero-duration stages
            has_zero_duration = any(
                st.duration_seconds == 0
                for st in exp_timing.stages.values()
            )

            if not has_zero_duration:
                continue

            # Use previous experiment's end time as this experiment's start
            if i > 0:
                prev_seed, prev_name, prev_timing = experiments[i - 1]

                # Get the end time of the previous experiment (last stage)
                prev_end_time = None
                for stage in reversed(stages):
                    if stage in prev_timing.stages:
                        prev_stage = prev_timing.stages[stage]
                        if prev_stage.end_time:
                            prev_end_time = prev_stage.end_time
                            break

                if prev_end_time:
                    # Set start time of first stage to previous experiment's end
                    first_stage_name = stages[0] if stages else None
                    if first_stage_name and first_stage_name in exp_timing.stages:
                        first_stage = exp_timing.stages[first_stage_name]

                        # Calculate new duration for first stage
                        if first_stage.end_time:
                            first_stage.start_time = prev_end_time
                            first_stage.duration_seconds = (
                                first_stage.end_time - first_stage.start_time
                            ).total_seconds()

                            # Update subsequent stages based on sequential execution
                            prev_stage_end = first_stage.end_time
                            for stage_name in stages[1:]:
                                if stage_name in exp_timing.stages:
                                    stage = exp_timing.stages[stage_name]
                                    if stage.end_time:
                                        stage.start_time = prev_stage_end
                                        stage.duration_seconds = (
                                            stage.end_time - stage.start_time
                                        ).total_seconds()
                                        prev_stage_end = stage.end_time

                            # Recalculate total duration
                            earliest = first_stage.start_time
                            latest = prev_stage_end
                            if earliest and latest:
                                exp_timing.total_duration_seconds = (
                                    latest - earliest
                                ).total_seconds()


def analyze_suite_timing(suite_path: Path) -> Dict[str, ExperimentTiming]:
    """Analyze timing for all experiments in a suite.

    Args:
        suite_path: Path to the suite directory containing experiment folders

    Returns:
        Dictionary mapping experiment names to their timing information
    """
    results: Dict[str, ExperimentTiming] = {}

    config = get_statistics_config()
    skip_dirs = set(config.skip_directories)

    for item in sorted(suite_path.iterdir()):
        if not item.is_dir() or item.name in skip_dirs:
            continue

        # Check if it looks like an experiment directory
        if any((item / stage).exists() for stage in config.stages):
            timing = analyze_experiment_timing(item)
            results[item.name] = timing

    # Apply sequential timing inference for copied data
    apply_sequential_timing_inference(results)

    return results


def summarize_timing(timings: Dict[str, ExperimentTiming]) -> Dict[str, Any]:
    """Generate summary statistics for a collection of experiment timings.

    Args:
        timings: Dictionary of experiment timings

    Returns:
        Dictionary with summary statistics
    """
    config = get_statistics_config()

    stage_durations: Dict[str, List[float]] = defaultdict(list)
    total_durations: List[float] = []

    for exp_timing in timings.values():
        for stage_name, stage_timing in exp_timing.stages.items():
            if stage_timing.duration_seconds is not None:
                stage_durations[stage_name].append(stage_timing.duration_seconds)

        if exp_timing.total_duration_seconds is not None:
            total_durations.append(exp_timing.total_duration_seconds)

    summary: Dict[str, Any] = {
        "num_experiments": len(timings),
        "stages": {},
        "total": {},
    }

    # Summarize per stage
    for stage in config.stages:
        durations = stage_durations.get(stage, [])
        if durations:
            summary["stages"][stage] = {
                "count": len(durations),
                "mean_seconds": mean(durations),
                "std_seconds": pstdev(durations) if len(durations) > 1 else 0.0,
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "mean_formatted": _format_duration(mean(durations)),
            }

    # Summarize total
    if total_durations:
        summary["total"] = {
            "count": len(total_durations),
            "mean_seconds": mean(total_durations),
            "std_seconds": pstdev(total_durations) if len(total_durations) > 1 else 0.0,
            "min_seconds": min(total_durations),
            "max_seconds": max(total_durations),
            "mean_formatted": _format_duration(mean(total_durations)),
            "total_sum_seconds": sum(total_durations),
            "total_sum_formatted": _format_duration(sum(total_durations)),
        }

    return summary


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def export_timing_report(
    timings: Dict[str, ExperimentTiming],
    output_path: Path,
    include_summary: bool = True,
) -> None:
    """Export timing analysis to JSON file.

    Args:
        timings: Dictionary of experiment timings
        output_path: Path to output JSON file
        include_summary: Whether to include summary statistics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "experiments": {
            name: timing.to_dict() for name, timing in timings.items()
        },
    }

    if include_summary:
        report["summary"] = summarize_timing(timings)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def export_latex_table(
    timings: Dict[str, ExperimentTiming],
    output_path: Path,
    caption: str = "Experiment execution times by stage.",
    label: str = "tab:timing",
) -> str:
    """Export timing analysis as a LaTeX table.

    Args:
        timings: Dictionary of experiment timings
        output_path: Path to output .tex file
        caption: Table caption
        label: Table label for referencing

    Returns:
        LaTeX table string
    """
    config = get_statistics_config()
    stages = config.stages

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")

    # Column specification: experiment name + stages + total
    col_spec = "l" + "r" * len(stages) + "r"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header_cols = [r"\textbf{Experiment}"]
    for stage in stages:
        header_cols.append(f"\\textbf{{{stage.capitalize()}}}")
    header_cols.append(r"\textbf{Total}")
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for exp_name, exp_timing in sorted(timings.items()):
        # Parse experiment name for cleaner display
        display_name = exp_name.replace("_", r"\_")

        row_cols = [display_name]
        for stage in stages:
            stage_timing = exp_timing.stages.get(stage)
            if stage_timing and stage_timing.duration_seconds is not None:
                row_cols.append(stage_timing.duration_formatted)
            else:
                row_cols.append("--")

        row_cols.append(exp_timing.total_duration_formatted)
        lines.append(" & ".join(row_cols) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_content = "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(latex_content)

    return latex_content


def export_latex_summary_table(
    timings: Dict[str, ExperimentTiming],
    output_path: Path,
    caption: str = "Summary of execution times across experiments.",
    label: str = "tab:timing_summary",
) -> str:
    """Export timing summary as a LaTeX table with statistics.

    Args:
        timings: Dictionary of experiment timings
        output_path: Path to output .tex file
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    summary = summarize_timing(timings)
    config = get_statistics_config()

    lines = []
    lines.append(r"% Requires: \usepackage{xcolor,colortbl}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Stage} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\")
    lines.append(r"\midrule")

    # Stage rows
    for stage in config.stages:
        if stage in summary.get("stages", {}):
            stats = summary["stages"][stage]
            # Add green background if std is 0 (perfect stability)
            row_color = r"\rowcolor{green!15} " if stats['std_seconds'] < 0.01 else ""
            lines.append(
                f"{row_color}{stage.capitalize()} & "
                f"{stats['mean_formatted']} & "
                f"{_format_duration(stats['std_seconds'])} & "
                f"{_format_duration(stats['min_seconds'])} & "
                f"{_format_duration(stats['max_seconds'])} \\\\"
            )

    lines.append(r"\midrule")

    # Total row
    if summary.get("total"):
        total = summary["total"]
        # Add green background if std is 0 (perfect stability)
        row_color = r"\rowcolor{green!15} " if total['std_seconds'] < 0.01 else ""
        lines.append(
            f"{row_color}\\textbf{{Total}} & "
            f"\\textbf{{{total['mean_formatted']}}} & "
            f"{_format_duration(total['std_seconds'])} & "
            f"{_format_duration(total['min_seconds'])} & "
            f"{_format_duration(total['max_seconds'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(latex_content)

    return latex_content


def generate_timing_plots(
    timings: Dict[str, ExperimentTiming],
    output_dir: Path,
    dpi: int = 150,
) -> List[Path]:
    """Generate timing visualization plots.

    Args:
        timings: Dictionary of experiment timings
        output_dir: Directory to save plots
        dpi: Plot resolution

    Returns:
        List of paths to generated plot files
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []
    config = get_statistics_config()
    stages = config.stages

    # Color palette
    colors = {
        "reduction": "#264653",
        "augmentation": "#e76f51",
        "total": "#2a9d8f",
    }

    # =========================================================================
    # Plot 1: Stacked bar chart of stage durations per experiment
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_names = sorted(timings.keys())
    x = range(len(exp_names))
    bar_width = 0.6

    # Collect data
    stage_data = {stage: [] for stage in stages}
    for exp_name in exp_names:
        exp_timing = timings[exp_name]
        for stage in stages:
            stage_timing = exp_timing.stages.get(stage)
            if stage_timing and stage_timing.duration_seconds:
                stage_data[stage].append(stage_timing.duration_seconds / 60)  # Convert to minutes
            else:
                stage_data[stage].append(0)

    # Plot stacked bars
    bottom = [0] * len(exp_names)
    for stage in stages:
        ax.bar(x, stage_data[stage], bar_width, bottom=bottom,
               label=stage.capitalize(), color=colors.get(stage, "#888888"))
        bottom = [b + d for b, d in zip(bottom, stage_data[stage])]

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Execution Time by Stage")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in exp_names], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "timing_stacked_bar.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    generated_files.append(plot_path)

    # =========================================================================
    # Plot 2: Box plot of stage durations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = []
    box_labels = []
    box_colors = []

    for stage in stages:
        durations = [stage_data[stage][i] for i in range(len(exp_names)) if stage_data[stage][i] > 0]
        if durations:
            box_data.append(durations)
            box_labels.append(stage.capitalize())
            box_colors.append(colors.get(stage, "#888888"))

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Duration (minutes)")
        ax.set_title("Distribution of Stage Durations")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "timing_boxplot.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        generated_files.append(plot_path)

    # =========================================================================
    # Plot 3: Pie chart of average time distribution
    # =========================================================================
    summary = summarize_timing(timings)

    if summary.get("stages"):
        fig, ax = plt.subplots(figsize=(7, 7))

        pie_data = []
        pie_labels = []
        pie_colors = []

        for stage in stages:
            if stage in summary["stages"]:
                mean_sec = summary["stages"][stage]["mean_seconds"]
                # Only include positive values (skip zero or negative durations)
                if mean_sec > 0:
                    pie_data.append(mean_sec)
                    pie_labels.append(f"{stage.capitalize()}\n({summary['stages'][stage]['mean_formatted']})")
                    pie_colors.append(colors.get(stage, "#888888"))

        if pie_data and len(pie_data) > 0:
            wedges, texts, autotexts = ax.pie(
                pie_data,
                labels=pie_labels,
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                explode=[0.02] * len(pie_data),
            )
            ax.set_title("Average Time Distribution by Stage")

            plt.tight_layout()
            plot_path = output_dir / "timing_pie.png"
            plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            generated_files.append(plot_path)

    # =========================================================================
    # Plot 4: Timeline/Gantt-style chart for a sample experiment
    # =========================================================================
    if timings:
        # Pick first experiment with complete timing
        sample_exp = None
        for exp_name, exp_timing in timings.items():
            if all(exp_timing.stages.get(s) and exp_timing.stages[s].start_time for s in stages):
                sample_exp = (exp_name, exp_timing)
                break

        if sample_exp:
            exp_name, exp_timing = sample_exp
            fig, ax = plt.subplots(figsize=(10, 3))

            y_pos = 0
            for i, stage in enumerate(stages):
                stage_timing = exp_timing.stages.get(stage)
                if stage_timing and stage_timing.start_time and stage_timing.end_time:
                    start = stage_timing.start_time
                    duration = stage_timing.duration_seconds / 60  # minutes

                    # Find earliest start for x-axis
                    earliest = min(
                        exp_timing.stages[s].start_time
                        for s in stages
                        if exp_timing.stages.get(s) and exp_timing.stages[s].start_time
                    )
                    x_start = (start - earliest).total_seconds() / 60

                    ax.barh(i, duration, left=x_start, height=0.5,
                            color=colors.get(stage, "#888888"), alpha=0.8,
                            label=stage.capitalize() if i == 0 else "")

                    # Add duration label
                    ax.text(x_start + duration / 2, i,
                            f"{stage_timing.duration_formatted}",
                            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

            ax.set_yticks(range(len(stages)))
            ax.set_yticklabels([s.capitalize() for s in stages])
            ax.set_xlabel("Time (minutes)")
            ax.set_title(f"Execution Timeline: {exp_name}")
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / "timing_timeline.png"
            plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            generated_files.append(plot_path)

    # =========================================================================
    # TREND CHARTS
    # =========================================================================

    # Helper: Extract reduction ratio from experiment name or metadata
    def extract_ratio(exp_name: str) -> float | None:
        """Extract reduction ratio from experiment name (e.g., dataset_01_03 -> 0.1)"""
        import re
        # Pattern: dataset_XX_YY where XX is ratio (01=0.1, 02=0.2, etc.)
        match = re.search(r'_(\d{2})_\d{2}$', exp_name)
        if match:
            ratio_int = int(match.group(1))
            return ratio_int / 10.0
        return None

    # =========================================================================
    # Plot 5: Reduction Ratio Trend
    # =========================================================================
    ratio_data = {}
    for exp_name, exp_timing in timings.items():
        ratio = extract_ratio(exp_name)
        if ratio is not None:
            if ratio not in ratio_data:
                ratio_data[ratio] = {stage: [] for stage in stages}
                ratio_data[ratio]['total'] = []

            for stage in stages:
                st = exp_timing.stages.get(stage)
                if st and st.duration_seconds:
                    ratio_data[ratio][stage].append(st.duration_seconds / 60)

            if exp_timing.total_duration_seconds:
                ratio_data[ratio]['total'].append(exp_timing.total_duration_seconds / 60)

    if ratio_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        ratios_sorted = sorted(ratio_data.keys())

        # Plot line for each stage
        for stage in stages:
            means = [mean(ratio_data[r][stage]) if ratio_data[r][stage] else 0
                    for r in ratios_sorted]
            if any(m > 0 for m in means):
                ax.plot(ratios_sorted, means, marker='o', linewidth=2,
                       label=stage.capitalize(), color=colors.get(stage, "#888888"))

        # Plot total line
        totals = [mean(ratio_data[r]['total']) if ratio_data[r]['total'] else 0
                 for r in ratios_sorted]
        if any(t > 0 for t in totals):
            ax.plot(ratios_sorted, totals, marker='s', linewidth=2.5,
                   label='Total', color=colors.get('total', '#2a9d8f'),
                   linestyle='--')

        ax.set_xlabel("Reduction Ratio", fontsize=11)
        ax.set_ylabel("Duration (minutes)", fontsize=11)
        ax.set_title("Execution Time vs Reduction Ratio", fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "timing_trend_ratio.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        generated_files.append(plot_path)

    # =========================================================================
    # Plot 6: Stage Duration Trend by Experiment
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_names_sorted = sorted(timings.keys())
    x = range(len(exp_names_sorted))

    for stage in stages:
        durations = []
        for exp_name in exp_names_sorted:
            st = timings[exp_name].stages.get(stage)
            if st and st.duration_seconds:
                durations.append(st.duration_seconds / 60)
            else:
                durations.append(0)

        if any(d > 0 for d in durations):
            ax.plot(x, durations, marker='o', linewidth=2,
                   label=stage.capitalize(), color=colors.get(stage, "#888888"))

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Duration (minutes)", fontsize=11)
    ax.set_title("Stage Duration Trend Across Experiments", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:20] for n in exp_names_sorted], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "timing_trend_experiments.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    generated_files.append(plot_path)

    # =========================================================================
    # Plot 7: Cumulative Time Trend
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    cumulative = 0
    cumulative_times = []

    for exp_name in exp_names_sorted:
        if timings[exp_name].total_duration_seconds:
            cumulative += timings[exp_name].total_duration_seconds / 3600  # hours
        cumulative_times.append(cumulative)

    ax.plot(x, cumulative_times, marker='o', linewidth=2.5,
           color='#2a9d8f', markersize=6)
    ax.fill_between(x, cumulative_times, alpha=0.3, color='#2a9d8f')

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Cumulative Time (hours)", fontsize=11)
    ax.set_title("Cumulative Execution Time", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:20] for n in exp_names_sorted], rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.3)

    # Add final total annotation
    if cumulative_times:
        ax.annotate(f'Total: {cumulative_times[-1]:.1f}h',
                   xy=(len(cumulative_times)-1, cumulative_times[-1]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = output_dir / "timing_trend_cumulative.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    generated_files.append(plot_path)

    # =========================================================================
    # Plot 8: Efficiency Trend (Time per 1000 entities)
    # =========================================================================
    # Try to load metadata to get entity counts
    efficiency_data = {}

    for exp_name, exp_timing in timings.items():
        ratio = extract_ratio(exp_name)
        if ratio is None:
            continue

        # Try to load metadata to get entity count
        metadata_path = exp_timing.experiment_path / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Get target entities from metadata
            target_entities = metadata.get("target_entities")
            if not target_entities:
                continue

            # Calculate efficiency for each stage
            for stage in stages:
                st = exp_timing.stages.get(stage)
                if st and st.duration_seconds and st.duration_seconds > 0:
                    # Seconds per 1000 entities
                    efficiency = (st.duration_seconds / target_entities) * 1000

                    if ratio not in efficiency_data:
                        efficiency_data[ratio] = {s: [] for s in stages}
                    efficiency_data[ratio][stage].append(efficiency)
        except Exception:
            continue

    if efficiency_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        ratios_sorted = sorted(efficiency_data.keys())

        for stage in stages:
            efficiencies = []
            for r in ratios_sorted:
                stage_effs = efficiency_data[r].get(stage, [])
                if stage_effs:
                    efficiencies.append(mean(stage_effs))
                else:
                    efficiencies.append(None)

            # Filter None values
            valid_ratios = [r for r, e in zip(ratios_sorted, efficiencies) if e is not None]
            valid_effs = [e for e in efficiencies if e is not None]

            if valid_effs:
                ax.plot(valid_ratios, valid_effs, marker='o', linewidth=2,
                       label=stage.capitalize(), color=colors.get(stage, "#888888"))

        ax.set_xlabel("Reduction Ratio", fontsize=11)
        ax.set_ylabel("Seconds per 1000 entities", fontsize=11)
        ax.set_title("Computational Efficiency Trend", fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "timing_trend_efficiency.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        generated_files.append(plot_path)

    # =========================================================================
    # Plot 9: Stage Proportion Trend (Stacked Area)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate proportions
    proportions = {stage: [] for stage in stages}

    for exp_name in exp_names_sorted:
        exp_timing = timings[exp_name]
        total = exp_timing.total_duration_seconds or 0

        if total > 0:
            for stage in stages:
                st = exp_timing.stages.get(stage)
                if st and st.duration_seconds:
                    prop = (st.duration_seconds / total) * 100
                    proportions[stage].append(prop)
                else:
                    proportions[stage].append(0)
        else:
            for stage in stages:
                proportions[stage].append(0)

    # Create stacked area
    ax.stackplot(x,
                [proportions[s] for s in stages],
                labels=[s.capitalize() for s in stages],
                colors=[colors.get(s, "#888888") for s in stages],
                alpha=0.7)

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Time Distribution (%)", fontsize=11)
    ax.set_title("Stage Time Proportion Trend", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n[:20] for n in exp_names_sorted], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "timing_trend_proportion.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    generated_files.append(plot_path)

    print(f"Generated {len(generated_files)} plots in {output_dir}")
    return generated_files


def print_timing_report(timings: Dict[str, ExperimentTiming]) -> None:
    """Print timing report to console.

    Args:
        timings: Dictionary of experiment timings
    """
    config = get_statistics_config()

    print("\n" + "=" * 80)
    print("TIMING ANALYSIS REPORT")
    print("=" * 80)

    # Per-experiment timings
    for exp_name, exp_timing in sorted(timings.items()):
        print(f"\n{exp_name}:")

        for stage in config.stages:
            stage_timing = exp_timing.stages.get(stage)
            if stage_timing:
                duration_str = stage_timing.duration_formatted
                print(f"  {stage:15s}: {duration_str}")
            else:
                print(f"  {stage:15s}: N/A")

        print(f"  {'TOTAL':15s}: {exp_timing.total_duration_formatted}")

    # Summary
    summary = summarize_timing(timings)

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print(f"\nExperiments analyzed: {summary['num_experiments']}")

    if summary.get("stages"):
        print("\nPer-stage averages:")
        for stage, stats in summary["stages"].items():
            print(f"  {stage:15s}: {stats['mean_formatted']} (±{_format_duration(stats['std_seconds'])})")

    if summary.get("total"):
        total_stats = summary["total"]
        print(f"\nTotal experiment time:")
        print(f"  Average:  {total_stats['mean_formatted']} (±{_format_duration(total_stats['std_seconds'])})")
        print(f"  Range:    {_format_duration(total_stats['min_seconds'])} - {_format_duration(total_stats['max_seconds'])}")
        print(f"  Sum:      {total_stats['total_sum_formatted']}")

    print("=" * 80)


def main():
    """Main entry point for CLI usage."""
    import argparse
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.config.loader import PROJECT_ROOT

    parser = argparse.ArgumentParser(
        description="Analyze timing for experiment runs based on folder metadata."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Experiment or suite directories to analyze. If omitted, scans results/.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save JSON report.",
    )
    parser.add_argument(
        "--latex",
        help="Path to save LaTeX table (.tex file).",
    )
    parser.add_argument(
        "--latex-summary",
        help="Path to save LaTeX summary table (.tex file).",
    )
    parser.add_argument(
        "--plots",
        help="Directory to save timing plots.",
    )
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results"),
        help="Root directory containing experiment runs.",
    )
    parser.add_argument(
        "--all-exports",
        help="Directory to save all exports (JSON, LaTeX, plots).",
    )

    args = parser.parse_args()

    # Discover experiments
    all_timings: Dict[str, ExperimentTiming] = {}

    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        paths = [Path(args.results_root)]

    for path in paths:
        if not path.exists():
            print(f"Warning: Path does not exist: {path}", file=sys.stderr)
            continue

        # Check if it's a single experiment or a suite
        config = get_statistics_config()
        if any((path / stage).exists() for stage in config.stages):
            # Single experiment
            timing = analyze_experiment_timing(path)
            all_timings[path.name] = timing
        else:
            # Suite directory - scan for experiments
            suite_timings = analyze_suite_timing(path)
            all_timings.update(suite_timings)

    if not all_timings:
        print("No experiments found.", file=sys.stderr)
        return 1

    # Print report
    print_timing_report(all_timings)

    # Handle --all-exports shortcut
    if args.all_exports:
        export_dir = Path(args.all_exports)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Set default paths for all exports
        if not args.output:
            args.output = str(export_dir / "timing_report.json")
        if not args.latex:
            args.latex = str(export_dir / "timing_table.tex")
        if not args.latex_summary:
            args.latex_summary = str(export_dir / "timing_summary.tex")
        if not args.plots:
            args.plots = str(export_dir / "plots")

    # Export JSON if requested
    if args.output:
        output_path = Path(args.output)
        export_timing_report(all_timings, output_path)
        print(f"\nJSON report saved to: {output_path}")

    # Export LaTeX table if requested
    if args.latex:
        latex_path = Path(args.latex)
        latex_content = export_latex_table(all_timings, latex_path)
        print(f"LaTeX table saved to: {latex_path}")
        print("\n--- LaTeX Table Preview ---")
        print(latex_content)

    # Export LaTeX summary table if requested
    if args.latex_summary:
        summary_path = Path(args.latex_summary)
        summary_content = export_latex_summary_table(all_timings, summary_path)
        print(f"\nLaTeX summary saved to: {summary_path}")
        print("\n--- LaTeX Summary Preview ---")
        print(summary_content)

    # Generate plots if requested
    if args.plots:
        plots_dir = Path(args.plots)
        generated = generate_timing_plots(all_timings, plots_dir)
        if generated:
            print(f"\nGenerated {len(generated)} plots:")
            for p in generated:
                print(f"  - {p}")

    return 0


if __name__ == "__main__":
    exit(main())
