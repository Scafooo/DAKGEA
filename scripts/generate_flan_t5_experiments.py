#!/usr/bin/env python3
"""Generate experiment configurations for FLAN-T5-XL augmentation.

This script creates configs for experiments using pre-trained FLAN-T5-XL models.

Generates configs matching the existing structure:
- flan_t5_bert_int/: 500 configs (5 datasets × 10 red × 10 aug ratios)

Output structure matches results/massive_aug_red_bert_int/ for statistics compatibility.

Usage:
    python scripts/generate_flan_t5_experiments.py
    python scripts/generate_flan_t5_experiments.py --dry-run
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "experiments" / "massive"

# Datasets for FLAN-T5 experiments
DATASETS = [
    ("BBC_DB", "openea/BBC_DB"),
    ("D_W_15K_V1", "openea/D_W_15K_V1"),
    ("D_W_15K_V2", "openea/D_W_15K_V2"),
    ("ICEWS_WIKI", "openea/ICEWS_WIKI"),
    ("ICEWS_YAGO", "openea/ICEWS_YAGO"),
]

# Ratios: 01-10 → 0.1-1.0
RATIOS = [(f"{i:02d}", round(i * 0.1, 1)) for i in range(1, 11)]

# Pre-trained models directory
PRETRAINED_DIR = "models/pretrained_plm"

# Seed (fixed across all experiments)
SEED = 11037

# Suite name (determines output directory in results/)
SUITE_NAME = "flan_t5_bert_int"


def create_experiment_config(
    dataset_name: str,
    dataset_path: str,
    reduction_ratio: float,
    augmentation_ratio: float,
    red_idx: str,
    aug_idx: str,
) -> Dict:
    """Create experiment configuration.

    Uses pre-trained FLAN-T5-XL model and skips fine-tuning.
    Both reduction and augmentation stages are evaluated.
    """
    pretrained_model = f"{PRETRAINED_DIR}/{dataset_name}"

    config = {
        "experiment": {
            "suite": SUITE_NAME,
            "name": f"{dataset_name}_{red_idx}_{aug_idx}",
            "dataset": {
                "name": dataset_path,
                "writer": "bert_int",
            },
            "reduction": {
                "method": "forget_labels",
                "ratio": reduction_ratio,
                "writer": "bert_int",
                "eval": True,  # Evaluate baseline
                "save_dataset": False,
                "save_model": False,
            },
            "augmentation": {
                "method": "plm_mixup",
                "ratio": augmentation_ratio,
                "backbone": "flan-t5-xl",
                "pretrained_model_dir": pretrained_model,
                "writer": {
                    "type": "bert_int",
                    "augmented_only_train": True,
                },
                "eval": True,
                "save_dataset": False,
                "save_model": False,
            },
            "model": "bert_int",
            "seed": SEED,
            "clear": True,
            "overwrite_existing": False,
        }
    }
    return config


def write_config(config: Dict, output_path: Path, dry_run: bool = False):
    """Write configuration to YAML file."""
    if dry_run:
        print(f"  [DRY-RUN] Would write: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_configs(dry_run: bool = False) -> int:
    """Generate all experiment configurations.

    Returns:
        Number of configs generated
    """
    config_count = 0
    output_dir = CONFIG_DIR / SUITE_NAME

    for dataset_name, dataset_path in DATASETS:
        print(f"\nDataset: {dataset_name}")

        # Generate configs (grid of reduction × augmentation)
        for red_idx, red_ratio in RATIOS:
            for aug_idx, aug_ratio in RATIOS:
                config = create_experiment_config(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    reduction_ratio=red_ratio,
                    augmentation_ratio=aug_ratio,
                    red_idx=red_idx,
                    aug_idx=aug_idx,
                )
                filename = f"{dataset_name}_{red_idx}_{aug_idx}.yaml"
                write_config(config, output_dir / filename, dry_run)
                config_count += 1

    return config_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate FLAN-T5 experiment configurations."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FLAN-T5 EXPERIMENT CONFIG GENERATOR")
    print("=" * 60)
    print(f"Datasets: {[d[0] for d in DATASETS]}")
    print(f"Reduction ratios: {[r[1] for r in RATIOS]}")
    print(f"Augmentation ratios: {[r[1] for r in RATIOS]}")
    print(f"Suite: {SUITE_NAME}")
    print(f"Output: {CONFIG_DIR / SUITE_NAME}")
    if args.dry_run:
        print("MODE: DRY-RUN")
    print("=" * 60)

    config_count = generate_configs(args.dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total configs: {config_count}")
    print(f"Results will be in: results/{SUITE_NAME}/")
    print("=" * 60)

    if not args.dry_run:
        print(f"\nConfigs written to: {CONFIG_DIR / SUITE_NAME}/")
        print("\nTo run experiments:")
        print("  # 1. Pre-train models (once)")
        print("  python scripts/pretrain_plm_per_dataset.py --all")
        print("")
        print("  # 2. Run experiments")
        print(f"  python -m experiments.runner config/experiments/massive/{SUITE_NAME}/")
        print("")
        print("  # 3. Analyze results")
        print(f"  python -m experiments.statistics.analyze_results results/{SUITE_NAME}/")


if __name__ == "__main__":
    main()
