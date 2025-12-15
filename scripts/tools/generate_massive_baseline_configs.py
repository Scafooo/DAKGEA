#!/usr/bin/env python3
"""Generate massive experiment configurations for baseline mode.

This script creates configs for Mode 1 (baseline - only real data) by removing
the augmentation section from the standard aug_red configs.

Generates configs for:
- bert_int_baseline/ (700 files)
- rrea_baseline/ (700 files)
"""

import yaml
from pathlib import Path
from typing import Dict, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "experiments" / "massive"

# Dataset configurations
DATASETS = [
    ("BBC_DB", "openea/BBC_DB"),
    ("D_W_15K_V1", "openea/D_W_15K_V1"),
    ("D_W_15K_V2", "openea/D_W_15K_V2"),
    ("ICEW_WIKI", "openea/ICEW_WIKI"),
    ("ICEW_YAGO", "openea/ICEW_YAGO"),
    ("SRPRS_D_W_15K_V1", "openea/SRPRS_D_W_15K_V1"),
    ("SRPRS_D_W_15K_V2", "openea/SRPRS_D_W_15K_V2"),
]

# Ratios: 01-10 → 0.1-1.0
REDUCTION_RATIOS = [(f"{i:02d}", i * 0.1) for i in range(1, 11)]
AUG_RATIOS = [(f"{i:02d}", i * 0.1) for i in range(1, 11)]  # Keep for naming consistency

# Model types
MODELS = ["bert_int", "rrea"]

# Seed (fixed across all experiments)
SEED = 11037


def create_baseline_config(
    dataset_name: str,
    dataset_path: str,
    reduction_ratio: float,
    model: str,
) -> Dict:
    """Create baseline experiment configuration.

    Args:
        dataset_name: Short dataset name (e.g., "BBC_DB")
        dataset_path: Full dataset path (e.g., "openea/BBC_DB")
        reduction_ratio: Data reduction ratio (0.1 to 1.0)
        model: Model type ("bert_int" or "rrea")

    Returns:
        Configuration dictionary
    """
    # Determine writer based on model
    # bert_int uses bert_int writer, rrea uses bert_int writer too (based on examples)
    writer = "bert_int"

    config = {
        "experiment": {
            "suite": f"massive_baseline_{model}",  # Group experiments by suite
            "name": f"{dataset_name}_baseline_{reduction_ratio:.1f}",
            "dataset": {
                "name": dataset_path,
                "writer": writer,
            },
            "reduction": {
                "method": "random_entities",
                "ratio": reduction_ratio,
                "writer": writer,
                "eval": True,
                "save_dataset": False,
                "save_model": False,
            },
            # NO augmentation section for baseline mode
            "model": model,
            "seed": SEED,
            "clear": True,
            "overwrite_existing": False,
        }
    }

    return config


def generate_configs_for_model(model: str) -> int:
    """Generate all baseline configs for a given model.

    Args:
        model: Model type ("bert_int" or "rrea")

    Returns:
        Number of files generated
    """
    output_dir = CONFIG_DIR / f"{model}_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for dataset_name, dataset_path in DATASETS:
        for red_idx, red_ratio in REDUCTION_RATIOS:
            for aug_idx, _ in AUG_RATIOS:  # Keep for naming consistency
                # File name: {DATASET}_{RED_IDX}_{AUG_IDX}.yaml
                # Even though we don't use augmentation, keep naming consistent
                filename = f"{dataset_name}_{red_idx}_{aug_idx}.yaml"
                filepath = output_dir / filename

                # Create config
                config = create_baseline_config(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    reduction_ratio=red_ratio,
                    model=model,
                )

                # Update experiment name to match filename pattern
                config["experiment"]["name"] = f"{dataset_name}_{red_idx}_{aug_idx}"

                # Write to file
                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                count += 1

    return count


def main():
    """Generate all baseline configurations."""
    print("=" * 80)
    print("Generating Baseline Experiment Configurations")
    print("=" * 80)
    print()

    total_files = 0

    for model in MODELS:
        print(f"Generating configs for {model}...")
        count = generate_configs_for_model(model)
        print(f"  ✓ Created {count} configs in config/experiments/massive/{model}_baseline/")
        total_files += count

    print()
    print("=" * 80)
    print(f"Total files generated: {total_files}")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Datasets: {len(DATASETS)}")
    print(f"  - Reduction ratios: {len(REDUCTION_RATIOS)} (0.1 to 1.0)")
    print(f"  - Aug ratio indices: {len(AUG_RATIOS)} (naming consistency)")
    print(f"  - Models: {len(MODELS)} (bert_int, rrea)")
    print(f"  - Total per model: {len(DATASETS) * len(REDUCTION_RATIOS) * len(AUG_RATIOS)}")
    print()
    print("Baseline configs created! Augmentation is DISABLED in these configs.")
    print()


if __name__ == "__main__":
    main()
