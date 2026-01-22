#!/usr/bin/env python3
"""Generate experiment configurations for Forget Labels mode.

This mode keeps the full knowledge graphs intact but hides a percentage of 
aligned entity labels (Forget Labels / Supervision Level).
"""

import yaml
from pathlib import Path
from typing import Dict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "experiments" / "massive" / "forget_labels"

# Dataset configurations
DATASETS = [
    ("BBC_DB", "openea/BBC_DB"),
    ("D_W_15K_V1", "openea/D_W_15K_V1"),
    ("D_W_15K_V2", "openea/D_W_15K_V2"),
    ("ICEW_WIKI", "openea/ICEW_WIKI"),
    ("ICEW_YAGO", "openea/ICEW_YAGO"),
]

# Ratios (Retention ratios): 0.1 to 1.0
# 0.1 means we KEEP 10% of labels and FORGET 90%
RATIOS = [(f"{i:02d}", round(i * 0.1, 1)) for i in range(1, 11)]

# Augmentation Ratios: Same as retention ratios usually
AUG_RATIOS = [(f"{i:02d}", round(i * 0.1, 1)) for i in range(1, 11)]

# Model
MODEL = "bert_int"

# Seed
SEED = 11037

def create_config(
    dataset_name: str,
    dataset_path: str,
    retention_ratio: float,
    aug_ratio: float,
    red_idx: str,
    aug_idx: str,
) -> Dict:
    writer = "bert_int"

    config = {
        "experiment": {
            "suite": f"forget_labels_{MODEL}",
            "name": f"{dataset_name}_{red_idx}_{aug_idx}",
            "dataset": {
                "name": dataset_path,
                "writer": writer,
            },
            "reduction": {
                "method": "forget_labels",  # Custom reducer
                "ratio": retention_ratio,   # Ratio of labels to KEEP
                "writer": writer,
                "eval": True,
                "save_dataset": False,
                "save_model": False,
            },
            "augmentation": {
                "method": "plm",
                "ratio": aug_ratio,
                "writer": {
                    "type": writer,
                    "augmented_only_train": True, # Fixed-order training Strategy B
                },
                "eval": True,
                "save_dataset": False,
                "save_model": False,
            },
            "model": MODEL,
            "seed": SEED,
            "clear": True,
            "overwrite_existing": False,
        }
    }
    return config

def main():
    print("Generating Forget Labels Configurations...")
    
    output_dir = CONFIG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for dataset_name, dataset_path in DATASETS:
        for red_idx, red_ratio in RATIOS:
            for aug_idx, aug_ratio in AUG_RATIOS:
                filename = f"{dataset_name}_{red_idx}_{aug_idx}.yaml"
                filepath = output_dir / filename
                
                config = create_config(dataset_name, dataset_path, red_ratio, aug_ratio, red_idx, aug_idx)
                
                with open(filepath, "w") as f:
                    yaml.dump(config, f, sort_keys=False)
                count += 1
                
    print(f"Generated {count} configs in {output_dir}")

if __name__ == "__main__":
    main()
