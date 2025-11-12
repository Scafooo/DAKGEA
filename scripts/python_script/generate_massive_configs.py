#!/usr/bin/env python3
"""Generate massive experiment configurations for PLM augmentation + reduction."""

import os
from pathlib import Path

# Base directory for experiments
BASE_DIR = Path("config/experiments/massive")
OUTPUT_DIR = BASE_DIR / "bert_int_aug_red"

# Dataset names (extracted from existing configs)
DATASETS = [
    "BBC_DB",
    "D_W_15K_V1",
    "D_W_15K_V2",
    "fr_en",
    "ICEW_WIKI",
    "ICEW_YAGO",
    "ja_en",
    "SRPRS_D_W_15K_V1",
    "SRPRS_D_W_15K_V2",
    "zh_en",
]

# Ratios for reduction and augmentation (0.1 to 1.0 with 0.1 intervals)
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Template for configuration file
CONFIG_TEMPLATE = """experiment:
  name: {name}
  dataset:
    name: openea/{dataset}
    writer: bert_int
  reduction:
    method: random_entities
    ratio: {reduction_ratio}
    writer: bert_int
    save: false
    eval: true
  augmentation:
    method: plm
    ratio: {augmentation_ratio}
    writer: bert_int
    save: false
    eval: true
  model: bert_int
  seed: 11037
  clear: true
  overwrite_existing: false
"""


def format_ratio(ratio: float) -> str:
    """Format ratio as two-digit string (e.g., 0.1 -> '01', 1.0 -> '10')."""
    return f"{int(ratio * 10):02d}"


def generate_configs():
    """Generate all configuration files."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0

    for dataset in DATASETS:
        for reduction_ratio in RATIOS:
            for augmentation_ratio in RATIOS:
                # Format ratios as two-digit strings
                red_str = format_ratio(reduction_ratio)
                aug_str = format_ratio(augmentation_ratio)

                # Create experiment name
                exp_name = f"{dataset}_{red_str}_{aug_str}"

                # Generate config content
                config_content = CONFIG_TEMPLATE.format(
                    name=exp_name,
                    dataset=dataset,
                    reduction_ratio=reduction_ratio,
                    augmentation_ratio=augmentation_ratio,
                )

                # Write config file
                output_file = OUTPUT_DIR / f"{exp_name}.yaml"
                with open(output_file, "w") as f:
                    f.write(config_content)

                total_files += 1

    print(f"✓ Generated {total_files} configuration files in {OUTPUT_DIR}")
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Reduction ratios: {len(RATIOS)} (0.1 to 1.0)")
    print(f"  Augmentation ratios: {len(RATIOS)} (0.1 to 1.0)")
    print(f"  Total combinations: {len(DATASETS)} × {len(RATIOS)} × {len(RATIOS)} = {total_files}")


if __name__ == "__main__":
    generate_configs()
