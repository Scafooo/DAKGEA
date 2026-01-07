#!/usr/bin/env python3
"""Generate LaTeX tables with experiment statistics.

⚠️ DEPRECATED: This standalone script is deprecated.
   Use `bash scripts/analyze_results.sh --export-formats latex` instead,
   which includes comparison tables generation as part of the full analysis.

For each dataset, creates a table with reduction ratios and metrics
comparing baseline (reduction only) vs augmented results.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_results(experiment_dir: Path) -> Dict:
    """Load results from an experiment directory.

    Returns:
        Dict with 'baseline' and augmentation results, or None if incomplete
    """
    results = {}

    # Load baseline (reduction) evaluation
    baseline_path = experiment_dir / "workspace" / "artifacts" / "evaluation" / "baseline" / "results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            results['baseline'] = json.load(f)

    # Load augmentation evaluation
    aug_path = experiment_dir / "workspace" / "artifacts" / "evaluation" / "plm" / "results.json"
    if aug_path.exists():
        with open(aug_path) as f:
            results['plm'] = json.load(f)

    return results if results else None


def parse_experiment_name(name: str) -> Tuple[str, float, int]:
    """Parse experiment name to extract dataset, ratio, seed.

    Example: "BBC_DB_05_03" -> ("BBC_DB", 0.5, 3)
    """
    parts = name.rsplit('_', 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid experiment name format: {name}")

    dataset = parts[0]
    ratio_idx = int(parts[1])
    seed_idx = int(parts[2])

    # Ratio mapping: 01->0.1, 02->0.2, ..., 10->1.0
    ratio = ratio_idx * 0.1

    return dataset, ratio, seed_idx


def aggregate_results(experiments_root: Path, model_name: str = "bert_int") -> Dict:
    """Aggregate results by dataset and ratio.

    Returns:
        Dict[dataset][ratio] = {
            'baseline': {'hits@1': [values], 'hits@10': [...], ...},
            'plm': {'hits@1': [values], ...}
        }
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: {
        'baseline': defaultdict(list),
        'plm': defaultdict(list)
    }))

    # Scan all experiment directories
    for exp_dir in experiments_root.iterdir():
        if not exp_dir.is_dir():
            continue

        try:
            dataset, ratio, seed_idx = parse_experiment_name(exp_dir.name)
        except ValueError:
            continue

        # Load results
        results = load_results(exp_dir)
        if not results:
            continue

        # Extract metrics for specified model
        for variant in ['baseline', 'plm']:
            if variant not in results:
                continue

            variant_results = results[variant]
            if model_name not in variant_results:
                continue

            metrics = variant_results[model_name]

            # Collect all metrics
            for metric in ['hits@1', 'hits@5', 'hits@10', 'mrr', 'mr', 'precision', 'recall', 'f-measure']:
                if metric in metrics:
                    aggregated[dataset][ratio][variant][metric].append(metrics[metric])

    return dict(aggregated)


def compute_statistics(values: List[float]) -> Tuple[float | None, float | None]:
    """Compute mean and std from list of values.

    Returns:
        Tuple of (mean, std) or (None, None) if no values available
    """
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def generate_latex_table(dataset: str, ratios_data: Dict, output_file: Path) -> None:
    """Generate LaTeX table for a single dataset with side-by-side comparison."""

    # Sort ratios
    sorted_ratios = sorted(ratios_data.keys())

    # Metrics to include
    metrics = [
        ('hits@1', 'H@1', True, 2, True),      # (key, header, is_percentage, decimals, higher_is_better)
        ('hits@5', 'H@5', True, 2, True),
        ('hits@10', 'H@10', True, 2, True),
        ('mrr', 'MRR', False, 4, True),
        ('mr', 'MR', False, 1, False),  # MR: lower is better
        ('precision', 'P', True, 2, True),
        ('recall', 'R', True, 2, True),
        ('f-measure', 'F1', True, 2, True),
    ]

    # Start LaTeX document
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")  # Use scriptsize for better fit
    latex.append(r"\caption{Results for " + dataset.replace("_", r"\_") + r"}")
    latex.append(r"\label{tab:" + dataset.lower() + r"}")

    # Build column specification: Ratio | Metric1 (Base/Aug) | Metric2 (Base/Aug) | ...
    # Each metric has 2 columns (baseline and augmented)
    num_metrics = len(metrics)
    col_spec = "c|" + "cc|" * num_metrics  # c for ratio, then pairs of cc for each metric
    latex.append(r"\begin{tabular}{" + col_spec + r"}")
    latex.append(r"\hline")

    # Header row 1: Metric names spanning 2 columns each
    header1 = r"\multirow{2}{*}{\textbf{Ratio}}"
    for metric_key, metric_name, _, _, _ in metrics:
        header1 += f" & \\multicolumn{{2}}{{c|}}{{{metric_name}}}"
    header1 += r" \\"
    latex.append(header1)

    # Header row 2: Base/Aug for each metric
    header2 = " "
    for _ in metrics:
        header2 += r" & \textit{Base} & \textit{Aug}"
    header2 += r" \\"
    latex.append(header2)
    latex.append(r"\hline")

    # Add data rows
    for ratio in sorted_ratios:
        data = ratios_data[ratio]

        # Start row with ratio
        row = f"\\textbf{{{ratio:.1f}}}"

        # For each metric, add baseline and augmented side by side
        for metric_key, _, is_percentage, decimals, higher_is_better in metrics:
            # Get baseline stats
            baseline_mean, baseline_std = compute_statistics(data['baseline'].get(metric_key, []))
            plm_mean, plm_std = compute_statistics(data['plm'].get(metric_key, []))

            # Baseline column (no color)
            if baseline_mean is None:
                row += r" & \textcolor{gray}{N/A}"
            else:
                # Apply percentage scaling
                if is_percentage:
                    baseline_mean *= 100
                    baseline_std *= 100
                row += f" & {baseline_mean:.{decimals}f}$\\pm${baseline_std:.{decimals}f}"

            # Augmented column (with color based on improvement)
            if plm_mean is None:
                row += r" & \textcolor{gray}{N/A}"
            else:
                # Apply percentage scaling
                if is_percentage:
                    plm_mean *= 100
                    plm_std *= 100

                # Determine if augmented is better than baseline
                # Only apply color if both baseline and augmented are available
                if baseline_mean is not None:
                    # Calculate relative difference
                    if baseline_mean != 0:
                        rel_diff = abs(plm_mean - baseline_mean) / abs(baseline_mean)
                    else:
                        rel_diff = abs(plm_mean - baseline_mean)

                    # If values are essentially equal (< 0.5% relative difference), use green (stability)
                    if rel_diff < 0.005:
                        color_cmd = r"\cellcolor{green!15}"  # Green for stability/consistency
                    else:
                        # Values differ significantly, check for improvement/degradation
                        if higher_is_better:
                            is_improvement = plm_mean > baseline_mean
                        else:  # For MR, lower is better
                            is_improvement = plm_mean < baseline_mean

                        # Choose color: green for improvement, red for degradation
                        if is_improvement:
                            color_cmd = r"\cellcolor{green!15}"  # Green for improvement
                        else:
                            color_cmd = r"\cellcolor{red!15}"    # Red for degradation
                else:
                    # No baseline to compare, no color
                    color_cmd = ""

                row += f" & {color_cmd}{plm_mean:.{decimals}f}$\\pm${plm_std:.{decimals}f}"

        row += r" \\"
        latex.append(row)

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"✓ Generated table for {dataset} -> {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LaTeX comparison tables for experiment results"
    )
    parser.add_argument(
        "experiments_dir",
        nargs="?",
        default=None,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="experiments/statistics/comparison_tables",
        help="Output directory for LaTeX tables (default: experiments/statistics/comparison_tables)"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="bert_int",
        help="Model name to extract results for (default: bert_int)"
    )

    args = parser.parse_args()

    # Auto-detect experiments directory if not provided
    if args.experiments_dir is None:
        # Try to find massive experiment directories
        possible_paths = [
            Path("results/experiments/massive/bert_int_aug_red"),
            Path("results/experiments/massive/rrea_aug_red"),
            Path("results/experiments"),
        ]

        experiments_root = None
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                experiments_root = path
                print(f"📂 Auto-detected experiments directory: {experiments_root}")
                break

        if experiments_root is None:
            print("❌ No experiments directory found!")
            print("   Please specify a directory or run some experiments first.")
            print(f"   Tried: {', '.join(str(p) for p in possible_paths)}")
            return 1
    else:
        experiments_root = Path(args.experiments_dir)

        # Check if experiments directory exists
        if not experiments_root.exists():
            print(f"❌ Experiments directory not found: {experiments_root}")
            print(f"   Run some experiments first!")
            return 1

    output_dir = Path(args.output_dir)
    model_name = args.model

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating results from: {experiments_root}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Aggregate results
    aggregated = aggregate_results(experiments_root, model_name)

    if not aggregated:
        print("❌ No results found!")
        print("   Make sure experiments have been run and results.json files exist.")
        return 1

    print(f"Found results for {len(aggregated)} datasets")
    print()

    # Generate LaTeX table for each dataset
    for dataset, ratios_data in sorted(aggregated.items()):
        output_file = output_dir / f"{dataset}.tex"
        generate_latex_table(dataset, ratios_data, output_file)

    print()
    print(f"✓ Generated {len(aggregated)} LaTeX tables in {output_dir}")
    print()
    print("To include in your LaTeX document:")
    for dataset in sorted(aggregated.keys()):
        print(f"  \\input{{{output_dir}/{dataset}.tex}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
