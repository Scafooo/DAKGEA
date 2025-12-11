#!/usr/bin/env python3
"""Test the LaTeX statistics generator with mock data.

⚠️ DEPRECATED: This test is for the deprecated standalone comparison_tables.py.
   The functionality is now integrated into analyze_results.py/exporters.py.
"""

import json
import tempfile
from pathlib import Path

# Mock data structure for testing
MOCK_RESULTS = {
    "bert_int": {
        "hits@1": 0.4523,
        "hits@5": 0.7245,
        "hits@10": 0.8567,
        "mrr": 0.5432,
        "mr": 12.3,
        "precision": 0.6234,
        "recall": 0.5891,
        "f-measure": 0.6058,
    }
}


def create_mock_experiment(exp_dir: Path, baseline_results: dict, plm_results: dict):
    """Create mock experiment directory with results."""
    # Create directory structure
    baseline_dir = exp_dir / "workspace" / "artifacts" / "evaluation" / "baseline"
    plm_dir = exp_dir / "workspace" / "artifacts" / "evaluation" / "plm"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    plm_dir.mkdir(parents=True, exist_ok=True)

    # Write results
    with open(baseline_dir / "results.json", 'w') as f:
        json.dump(baseline_results, f)

    with open(plm_dir / "results.json", 'w') as f:
        json.dump(plm_results, f)


def main():
    """Create mock experiments and test the generator."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from experiments.statistics.comparison_tables import (
        aggregate_results,
        generate_latex_table,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_root = Path(tmpdir) / "experiments"
        exp_root.mkdir()

        print("Creating mock experiments...")

        # Create mock experiments for BBC_DB dataset with 3 ratios and 2 seeds each
        for ratio_idx in [1, 5, 10]:  # 0.1, 0.5, 1.0
            for seed_idx in [1, 2]:
                exp_name = f"BBC_DB_{ratio_idx:02d}_{seed_idx:02d}"
                exp_dir = exp_root / exp_name

                # Vary results slightly based on ratio and seed
                baseline_metrics = MOCK_RESULTS["bert_int"].copy()
                plm_metrics = MOCK_RESULTS["bert_int"].copy()

                # Baseline gets better with more data (higher ratio)
                baseline_metrics["hits@1"] += ratio_idx * 0.02
                baseline_metrics["hits@10"] += ratio_idx * 0.01

                # Augmentation improves over baseline
                plm_metrics["hits@1"] = baseline_metrics["hits@1"] + 0.05
                plm_metrics["hits@10"] = baseline_metrics["hits@10"] + 0.03

                # Add small seed variation
                import random
                random.seed(seed_idx)
                for key in baseline_metrics:
                    if isinstance(baseline_metrics[key], float):
                        baseline_metrics[key] += random.uniform(-0.01, 0.01)
                        plm_metrics[key] += random.uniform(-0.01, 0.01)

                create_mock_experiment(
                    exp_dir,
                    {"bert_int": baseline_metrics},
                    {"bert_int": plm_metrics}
                )

        print(f"✓ Created mock experiments in {exp_root}")
        print()

        # Test aggregation
        print("Aggregating results...")
        aggregated = aggregate_results(exp_root, "bert_int")

        print(f"✓ Found {len(aggregated)} datasets")
        for dataset, ratios in aggregated.items():
            print(f"  {dataset}: {len(ratios)} ratios")
        print()

        # Test LaTeX generation
        print("Generating LaTeX tables...")
        output_dir = Path(tmpdir) / "latex_tables"
        output_dir.mkdir()

        for dataset, ratios_data in aggregated.items():
            output_file = output_dir / f"{dataset}.tex"
            generate_latex_table(dataset, ratios_data, output_file)

        print()
        print("Generated LaTeX table:")
        print("=" * 80)
        with open(output_dir / "BBC_DB.tex") as f:
            print(f.read())
        print("=" * 80)
        print()
        print("✓ Test successful!")


if __name__ == "__main__":
    main()
