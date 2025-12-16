#!/usr/bin/env python3
"""
Script to separate multilingual configs from massive/multilingual/ into proper directories
for bert_int and rrea in formats: baseline, aug_red, synthetic_only
"""

import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent


def create_baseline_config(original_config, model_name, dataset_name):
    """Create baseline config (reduction only, no augmentation)"""
    config = {
        "experiment": {
            "suite": f"massive_baseline_{model_name}",
            "name": original_config["experiment"]["name"],
            "dataset": {
                "name": dataset_name,
                "writer": "bert_int"  # Always bert_int for writer
            },
            "reduction": original_config["experiment"]["reduction"].copy(),
            "model": model_name,
            "seed": original_config["experiment"].get("seed", 11037),
            "clear": original_config["experiment"].get("clear", True),
            "overwrite_existing": original_config["experiment"].get("overwrite_existing", False)
        }
    }
    return config


def create_aug_red_config(original_config, model_name, dataset_name):
    """Create aug_red config (reduction + augmentation with both original and synthetic)"""
    config = {
        "experiment": {
            "suite": f"massive_aug_red_{model_name}",
            "name": original_config["experiment"]["name"],
            "dataset": {
                "name": dataset_name,
                "writer": "bert_int"
            },
            "reduction": original_config["experiment"]["reduction"].copy(),
            "augmentation": {
                "method": original_config["experiment"]["augmentation"]["method"],
                "ratio": original_config["experiment"]["augmentation"].get("ratio", 0.1),
                "writer": {
                    "type": "bert_int"
                },
                "eval": original_config["experiment"]["augmentation"].get("eval", True),
                "save_dataset": original_config["experiment"]["augmentation"].get("save_dataset", False),
                "save_model": original_config["experiment"]["augmentation"].get("save_model", False)
            },
            "model": model_name,
            "seed": original_config["experiment"].get("seed", 11037),
            "clear": original_config["experiment"].get("clear", True),
            "overwrite_existing": original_config["experiment"].get("overwrite_existing", False)
        }
    }
    return config


def create_synthetic_only_config(original_config, model_name, dataset_name):
    """Create synthetic_only config (reduction + augmentation with only synthetic data)"""
    config = {
        "experiment": {
            "suite": f"massive_synthetic_only_{model_name}",
            "name": original_config["experiment"]["name"],
            "dataset": {
                "name": dataset_name,
                "writer": "bert_int"
            },
            "reduction": original_config["experiment"]["reduction"].copy(),
            "augmentation": {
                "method": original_config["experiment"]["augmentation"]["method"],
                "ratio": original_config["experiment"]["augmentation"].get("ratio", 0.1),
                "training_mode": "synthetic_only",
                "writer": {
                    "type": "bert_int"
                },
                "eval": original_config["experiment"]["augmentation"].get("eval", True),
                "save_dataset": original_config["experiment"]["augmentation"].get("save_dataset", False),
                "save_model": original_config["experiment"]["augmentation"].get("save_model", False)
            },
            "model": model_name,
            "seed": original_config["experiment"].get("seed", 11037),
            "clear": original_config["experiment"].get("clear", True),
            "overwrite_existing": original_config["experiment"].get("overwrite_existing", False)
        }
    }
    return config


def process_multilingual_config(config_path: Path):
    """Process a single multilingual config and create separate configs"""
    with open(config_path) as f:
        original_config = yaml.safe_load(f)

    # Extract dataset name from config
    dataset_name = original_config["experiment"]["dataset"]["name"]
    config_name = config_path.stem  # e.g., "zh_en_01_00"

    # Determine if this is a baseline (stub) or augmented (plm) config
    augmentation = original_config["experiment"].get("augmentation", {})
    aug_method = augmentation.get("method", "stub")

    results = []

    # Create configs for both bert_int and rrea
    for model_name in ["bert_int", "rrea"]:
        if aug_method == "stub":
            # This is a baseline config - only create baseline
            config = create_baseline_config(original_config, model_name, dataset_name)
            target_dir = PROJECT_ROOT / "config" / "experiments" / "massive" / f"{model_name}_baseline"
            results.append((config, target_dir / f"{config_name}.yaml"))
        elif aug_method == "plm":
            # This is an augmented config - create both aug_red and synthetic_only

            # Aug_red version
            aug_red_config = create_aug_red_config(original_config, model_name, dataset_name)
            target_dir_aug = PROJECT_ROOT / "config" / "experiments" / "massive" / f"{model_name}_aug_red"
            results.append((aug_red_config, target_dir_aug / f"{config_name}.yaml"))

            # Synthetic_only version
            synth_config = create_synthetic_only_config(original_config, model_name, dataset_name)
            target_dir_synth = PROJECT_ROOT / "config" / "experiments" / "massive" / f"{model_name}_synthetic_only"
            results.append((synth_config, target_dir_synth / f"{config_name}.yaml"))

    return results


def main():
    multilingual_dir = PROJECT_ROOT / "config" / "experiments" / "massive" / "multilingual"

    # Get all multilingual configs
    config_files = list(multilingual_dir.glob("*.yaml"))

    print(f"Found {len(config_files)} multilingual configs")

    # Create target directories if they don't exist
    for model in ["bert_int", "rrea"]:
        for variant in ["baseline", "aug_red", "synthetic_only"]:
            target_dir = PROJECT_ROOT / "config" / "experiments" / "massive" / f"{model}_{variant}"
            target_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    created = 0
    errors = 0

    for config_file in sorted(config_files):
        try:
            results = process_multilingual_config(config_file)

            for config, target_path in results:
                # Write the config
                with open(target_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                created += 1

            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(config_files)} files... (created {created} configs)")
        except Exception as e:
            print(f"Error processing {config_file}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nCompleted!")
    print(f"  Processed: {processed} multilingual configs")
    print(f"  Created: {created} separated configs")
    print(f"  Errors: {errors}")

    # Show summary by type
    print(f"\nBreakdown:")
    for model in ["bert_int", "rrea"]:
        for variant in ["baseline", "aug_red", "synthetic_only"]:
            target_dir = PROJECT_ROOT / "config" / "experiments" / "massive" / f"{model}_{variant}"
            count = len(list(target_dir.glob("*_en_*.yaml")))
            print(f"  {model}_{variant}: {count} configs")


if __name__ == "__main__":
    main()
