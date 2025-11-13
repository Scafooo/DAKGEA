#!/usr/bin/env python3
"""Update massive experiment configs to use granular save flags.

Converts legacy 'save' flag to 'save_dataset' and 'save_model' flags.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def update_config(data: Dict[str, Any]) -> bool:
    """Update a config dict to use granular save flags.

    Returns:
        True if changes were made, False otherwise
    """
    changed = False
    experiment = data.get("experiment", {})

    # Update reduction section
    if "reduction" in experiment:
        reduction = experiment["reduction"]
        if "save" in reduction:
            save_value = reduction.pop("save")
            reduction["save_dataset"] = save_value
            reduction["save_model"] = save_value
            changed = True

    # Update augmentation section
    if "augmentation" in experiment:
        augmentation = experiment["augmentation"]
        if "save" in augmentation:
            save_value = augmentation.pop("save")
            augmentation["save_dataset"] = save_value
            augmentation["save_model"] = save_value
            changed = True

    return changed


def main():
    """Update all massive experiment configs."""
    base_path = Path("config/experiments/massive")

    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return

    # Find all YAML files
    yaml_files = list(base_path.rglob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files to process")

    updated_count = 0
    skipped_count = 0

    for yaml_file in yaml_files:
        try:
            # Read the file
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Update the config
            if update_config(data):
                # Write back to file
                with open(yaml_file, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

                updated_count += 1
                print(f"✓ Updated: {yaml_file.relative_to(base_path)}")
            else:
                skipped_count += 1

        except Exception as e:
            print(f"✗ Error processing {yaml_file}: {e}")

    print(f"\nSummary:")
    print(f"  Updated: {updated_count} files")
    print(f"  Skipped: {skipped_count} files (no changes needed)")
    print(f"  Total: {len(yaml_files)} files")


if __name__ == "__main__":
    main()
