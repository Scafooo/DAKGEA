#!/usr/bin/env python3
"""Generate RREA experiment configuration files."""

from pathlib import Path

# Configuration template
CONFIG_TEMPLATE = """experiment:
  name: {name}
  dataset:
    name: {dataset}
    writer: bert_int
  reduction:
    method: random_entities
    ratio: {ratio}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  augmentation:
    method: plm
    ratio: {ratio}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  model: rrea
  seed: {seed}
  clear: true
  overwrite_existing: false
"""

# Dataset configurations
DATASETS = [
    "openea/BBC_DB",
    "openea/D_W_15K_V1",
    "openea/D_W_15K_V2",
    "openea/ICEW_WIKI",
    "openea/ICEW_YAGO",
    "openea/SRPRS_D_W_15K_V1",
    "openea/SRPRS_D_W_15K_V2",
]

# Ratios (0.1 to 1.0 in steps of 0.1)
RATIOS = [round(0.1 * i, 1) for i in range(1, 11)]

# Seeds (11037 to 11136 for 10 different seeds)
BASE_SEED = 11037
SEEDS = [BASE_SEED + i * 10 for i in range(10)]

# Output directory
OUTPUT_DIR = Path("config/experiments/massive/rrea_aug_red")


def generate_configs():
    """Generate all RREA configuration files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for dataset in DATASETS:
        # Extract dataset short name
        dataset_short = dataset.split("/")[1]

        for ratio_idx, ratio in enumerate(RATIOS, start=1):
            for seed_idx, seed in enumerate(SEEDS, start=1):
                # Create experiment name
                exp_name = f"{dataset_short}_{ratio_idx:02d}_{seed_idx:02d}"

                # Generate config content
                config_content = CONFIG_TEMPLATE.format(
                    name=exp_name,
                    dataset=dataset,
                    ratio=ratio,
                    seed=seed
                )

                # Write to file
                output_file = OUTPUT_DIR / f"{exp_name}.yaml"
                output_file.write_text(config_content)
                file_count += 1

    print(f"✓ Generated {file_count} RREA configuration files in {OUTPUT_DIR}")
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Ratios: {len(RATIOS)} (0.1 to 1.0)")
    print(f"  Seeds: {len(SEEDS)} (10 different seeds)")


if __name__ == "__main__":
    generate_configs()
