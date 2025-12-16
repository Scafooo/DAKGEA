#!/usr/bin/env python3
"""
Script to remove augmented_only_train flag from all experiment configs
"""

import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent


def remove_augmented_only_train_from_dict(data):
    """Recursively remove augmented_only_train from a dictionary"""
    if isinstance(data, dict):
        # Remove augmented_only_train if it exists
        if 'augmented_only_train' in data:
            del data['augmented_only_train']

        # Recursively process all values
        for key, value in list(data.items()):
            if isinstance(value, dict):
                remove_augmented_only_train_from_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        remove_augmented_only_train_from_dict(item)

    return data


def process_config_file(file_path):
    """Process a single config file and remove augmented_only_train"""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)

        # Remove augmented_only_train
        modified = remove_augmented_only_train_from_dict(config)

        # Write back
        with open(file_path, 'w') as f:
            yaml.dump(modified, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    # Find all config files
    config_dir = PROJECT_ROOT / "config" / "experiments"

    # Get all yaml files recursively
    config_files = list(config_dir.rglob("*.yaml"))

    print(f"Found {len(config_files)} config files")

    processed = 0
    errors = 0

    for config_file in config_files:
        if process_config_file(config_file):
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(config_files)} files...")
        else:
            errors += 1

    print(f"\nCompleted!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
