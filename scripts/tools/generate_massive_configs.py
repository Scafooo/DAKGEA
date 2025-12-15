#!/usr/bin/env python3
"""Generate massive experiment configurations for PLM augmentation + reduction.

This script creates configs for Mode 3 (augmented - real data + synthetic data).

Generates configs for:
- bert_int_aug_red/ (1000 files: 10 datasets × 10 red ratios × 10 aug ratios)
- rrea_aug_red/ (1000 files: 10 datasets × 10 red ratios × 10 aug ratios)
"""

import yaml
from pathlib import Path
from typing import Dict, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "experiments" / "massive"

# Dataset names (extracted from existing configs)
DATASETS = [
    ("BBC_DB", "openea/BBC_DB"),
    ("D_W_15K_V1", "openea/D_W_15K_V1"),
    ("D_W_15K_V2", "openea/D_W_15K_V2"),
    ("fr_en", "openea/fr_en"),
    ("ICEW_WIKI", "openea/ICEW_WIKI"),
    ("ICEW_YAGO", "openea/ICEW_YAGO"),
    ("ja_en", "openea/ja_en"),
    ("SRPRS_D_W_15K_V1", "openea/SRPRS_D_W_15K_V1"),
    ("SRPRS_D_W_15K_V2", "openea/SRPRS_D_W_15K_V2"),
    ("zh_en", "openea/zh_en"),
]

# Ratios: 01-10 → 0.1-1.0
RATIOS = [(f"{i:02d}", i * 0.1) for i in range(1, 11)]

# Model types
MODELS = ["bert_int", "rrea"]

# Seed (fixed across all experiments)
SEED = 11037


def create_augmented_config(
    dataset_name: str,
    dataset_path: str,
    reduction_ratio: float,
    augmentation_ratio: float,
    model: str,
) -> Dict:
    """Create augmented experiment configuration.

    Args:
        dataset_name: Short dataset name (e.g., "BBC_DB")
        dataset_path: Full dataset path (e.g., "openea/BBC_DB")
        reduction_ratio: Data reduction ratio (0.1 to 1.0)
        augmentation_ratio: Data augmentation ratio (0.1 to 1.0)
        model: Model type ("bert_int" or "rrea")

    Returns:
        Configuration dictionary
    """
    # Determine writer based on model
    writer = "bert_int"

    config = {
        "experiment": {
            "suite": f"massive_aug_red_{model}",  # Group experiments by suite
            "name": f"{dataset_name}_{reduction_ratio:.1f}_{augmentation_ratio:.1f}",
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
            "augmentation": {
                "method": "plm",
                "ratio": augmentation_ratio,
                "writer": {
                    "type": writer,
                    "augmented_only_train": True,
                },
                "eval": True,
                "save_dataset": False,
                "save_model": False,
            },
            "model": model,
            "seed": SEED,
            "clear": True,
            "overwrite_existing": False,
        }
    }

    return config


def format_ratio(ratio: float) -> str:
    """Format ratio as two-digit string (e.g., 0.1 -> '01', 1.0 -> '10')."""
    return f"{int(ratio * 10):02d}"


def generate_configs_for_model(model: str) -> int:
    """Generate all augmented configs for a given model.

    Args:
        model: Model type ("bert_int" or "rrea")

    Returns:
        Number of files generated
    """
    output_dir = CONFIG_DIR / f"{model}_aug_red"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for dataset_name, dataset_path in DATASETS:
        for red_idx, red_ratio in RATIOS:
            for aug_idx, aug_ratio in RATIOS:
                # File name: {DATASET}_{RED_IDX}_{AUG_IDX}.yaml
                filename = f"{dataset_name}_{red_idx}_{aug_idx}.yaml"
                filepath = output_dir / filename

                # Create config
                config = create_augmented_config(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    reduction_ratio=red_ratio,
                    augmentation_ratio=aug_ratio,
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
    """Generate all augmented configurations."""
    print("=" * 80)
    print("Generating Augmented Experiment Configurations")
    print("=" * 80)
    print()

    total_files = 0

    for model in MODELS:
        print(f"Generating configs for {model}...")
        count = generate_configs_for_model(model)
        total_files += count
        print(f"  ✓ Created {count} configs in {CONFIG_DIR / f'{model}_aug_red'}/")

    print()
    print("=" * 80)
    print(f"Total files generated: {total_files}")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Datasets: {len(DATASETS)}")
    print(f"  - Reduction ratios: {len(RATIOS)} (0.1 to 1.0)")
    print(f"  - Augmentation ratios: {len(RATIOS)} (0.1 to 1.0)")
    print(f"  - Models: {len(MODELS)} (bert_int, rrea)")
    print(f"  - Total per model: {len(DATASETS) * len(RATIOS) * len(RATIOS)}")
    print()
    print("Augmented configs created! These configs train with BOTH real and synthetic data.")


if __name__ == "__main__":
    main()
