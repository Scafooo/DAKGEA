"""Run experimental comparison: Baseline vs Synthetic-only vs Augmented.

This script runs three experiments to evaluate:
1. Baseline: Training with only original (real) aligned pairs
2. Synthetic-only: Training with ONLY augmented (synthetic) pairs
3. Augmented: Training with original + synthetic pairs (standard augmentation)

The goal is to answer:
- Can synthetic data replace real data? (synthetic_only vs baseline)
- How much benefit does augmentation provide? (augmented vs baseline)
- Is there quality gap in synthetic data? (baseline - synthetic_only)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger

logger = get_logger(__name__)


EXPERIMENTS = {
    "baseline": "config/experiments/synthetic_comparison/baseline.yaml",
    "synthetic_only": "config/experiments/synthetic_comparison/synthetic_only.yaml",
    "augmented": "config/experiments/synthetic_comparison/augmented.yaml",
}


def run_experiment(name: str, config_path: str, dry_run: bool = False) -> Dict:
    """Run a single experiment.

    Args:
        name: Experiment name
        config_path: Path to config file
        dry_run: If True, don't actually run, just print command

    Returns:
        Dictionary with results
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Running experiment: {name}")
    logger.info(f"  Config: {config_path}")

    # TODO: Replace with actual training command
    # This is placeholder - you'll need to integrate with your training pipeline
    cmd = [
        "python", "-m", "src.main",  # Adjust to your actual entry point
        "--config", config_path,
    ]

    logger.info(f"  Command: {' '.join(cmd)}")

    if dry_run:
        logger.info(f"  [DRY RUN] Skipping actual execution")
        return {"name": name, "status": "dry_run"}

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"  ✓ Experiment {name} completed successfully")

        # TODO: Parse results from output or results file
        # This is placeholder
        return {
            "name": name,
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ Experiment {name} failed!")
        logger.error(f"    Error: {e}")
        return {
            "name": name,
            "status": "failed",
            "error": str(e),
        }


def load_results(results_dir: str) -> Dict:
    """Load results from a completed experiment.

    Args:
        results_dir: Directory containing results.json

    Returns:
        Dictionary with metrics
    """
    results_file = Path(results_dir) / "results.json"

    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}")
        return {}

    with open(results_file, "r") as f:
        return json.load(f)


def compare_results(results: Dict[str, Dict]) -> None:
    """Compare results across experiments and print analysis.

    Args:
        results: Dictionary mapping experiment name to results
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENTAL COMPARISON RESULTS")
    logger.info("=" * 80)

    # Extract key metrics (adjust based on your actual metrics)
    metrics_to_compare = ["hits@1", "hits@10", "mrr"]

    # Print table header
    print(f"\n{'Metric':<15} {'Baseline':>12} {'Synthetic':>12} {'Augmented':>12} {'Synth Gap':>12} {'Aug Benefit':>12}")
    print("-" * 80)

    for metric in metrics_to_compare:
        baseline_val = results.get("baseline", {}).get(metric, 0.0)
        synthetic_val = results.get("synthetic_only", {}).get(metric, 0.0)
        augmented_val = results.get("augmented", {}).get(metric, 0.0)

        # Compute comparisons
        synth_gap = baseline_val - synthetic_val  # Quality gap
        aug_benefit = augmented_val - baseline_val  # Augmentation benefit

        print(f"{metric:<15} {baseline_val:>12.4f} {synthetic_val:>12.4f} {augmented_val:>12.4f} {synth_gap:>12.4f} {aug_benefit:>12.4f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    baseline_hits1 = results.get("baseline", {}).get("hits@1", 0.0)
    synthetic_hits1 = results.get("synthetic_only", {}).get("hits@1", 0.0)
    augmented_hits1 = results.get("augmented", {}).get("hits@1", 0.0)

    quality_gap_pct = ((baseline_hits1 - synthetic_hits1) / baseline_hits1 * 100) if baseline_hits1 > 0 else 0
    aug_benefit_pct = ((augmented_hits1 - baseline_hits1) / baseline_hits1 * 100) if baseline_hits1 > 0 else 0

    print(f"\n1. Quality Gap (Baseline - Synthetic): {quality_gap_pct:.2f}%")
    if quality_gap_pct < 5:
        print("   → Synthetic data has EXCELLENT quality (< 5% gap)")
    elif quality_gap_pct < 15:
        print("   → Synthetic data has GOOD quality (5-15% gap)")
    else:
        print("   → Synthetic data has SIGNIFICANT quality issues (> 15% gap)")

    print(f"\n2. Augmentation Benefit (Augmented - Baseline): {aug_benefit_pct:.2f}%")
    if aug_benefit_pct > 10:
        print("   → Augmentation provides STRONG benefit (> 10% improvement)")
    elif aug_benefit_pct > 5:
        print("   → Augmentation provides MODERATE benefit (5-10% improvement)")
    elif aug_benefit_pct > 0:
        print("   → Augmentation provides SMALL benefit (< 5% improvement)")
    else:
        print("   → Augmentation provides NO benefit or HURTS performance")

    print(f"\n3. Transferability Score (Synthetic / Baseline): {synthetic_hits1 / baseline_hits1:.3f}" if baseline_hits1 > 0 else "N/A")
    if synthetic_hits1 > baseline_hits1:
        print("   ⚠ WARNING: Synthetic > Baseline suggests potential ARTIFACTS/SHORTCUTS")
    elif synthetic_hits1 >= 0.95 * baseline_hits1:
        print("   ✓ Synthetic data transfers well (> 95% of baseline)")
    else:
        print("   → Synthetic data has transfer gap")


def main():
    parser = argparse.ArgumentParser(description="Run synthetic vs real data comparison experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()), default=list(EXPERIMENTS.keys()),
                        help="Which experiments to run (default: all)")
    parser.add_argument("--compare-only", action="store_true", help="Skip running, only compare existing results")
    args = parser.parse_args()

    results = {}

    if not args.compare_only:
        # Run experiments
        for exp_name in args.experiments:
            config_path = EXPERIMENTS[exp_name]
            result = run_experiment(exp_name, config_path, dry_run=args.dry_run)
            results[exp_name] = result

        logger.info("\n" + "=" * 80)
        logger.info("All experiments completed!")
        logger.info("=" * 80)

    # Load and compare results
    if not args.dry_run:
        logger.info("\nLoading results for comparison...")

        for exp_name in args.experiments:
            # Extract results_dir from config (placeholder)
            results_dir = f"results/synthetic_comparison/{exp_name}"
            results[exp_name] = load_results(results_dir)

        if any(results.values()):
            compare_results(results)
        else:
            logger.warning("No results found to compare. Have the experiments completed?")


if __name__ == "__main__":
    main()
