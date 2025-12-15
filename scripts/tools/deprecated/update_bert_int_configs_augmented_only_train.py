#!/usr/bin/env python3
"""Update all bert_int_aug_red config files to enable augmented_only_train option."""

import yaml
from pathlib import Path

CONFIG_DIR = Path("config/experiments/massive/bert_int_aug_red")


def update_config_file(config_path: Path) -> bool:
    """Update a single config file to enable augmented_only_train.

    Args:
        config_path: Path to YAML config file

    Returns:
        True if file was modified, False otherwise
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check if augmentation section exists
    if 'experiment' not in config or 'augmentation' not in config['experiment']:
        return False

    augmentation = config['experiment']['augmentation']

    # Check if writer is a string (needs to be converted to dict)
    if 'writer' not in augmentation:
        return False

    writer = augmentation['writer']

    # If writer is already a dict with augmented_only_train=true, skip
    if isinstance(writer, dict):
        if writer.get('augmented_only_train') == True:
            return False  # Already set, no change needed

    # Convert writer to dict format with augmented_only_train enabled
    if isinstance(writer, str):
        augmentation['writer'] = {
            'type': writer,
            'augmented_only_train': True
        }
    elif isinstance(writer, dict):
        # Writer is already a dict, just add the option
        writer['augmented_only_train'] = True

    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return True


def main():
    """Update all bert_int_aug_red config files."""
    if not CONFIG_DIR.exists():
        print(f"❌ Config directory not found: {CONFIG_DIR}")
        return

    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    if not config_files:
        print(f"❌ No config files found in {CONFIG_DIR}")
        return

    print(f"Found {len(config_files)} config files")
    print(f"Updating augmentation.writer to enable augmented_only_train...")

    modified_count = 0
    skipped_count = 0

    for config_path in config_files:
        try:
            if update_config_file(config_path):
                modified_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"❌ Failed to update {config_path.name}: {e}")
            continue

    print(f"\n✓ Update complete!")
    print(f"  Modified: {modified_count} files")
    print(f"  Skipped: {skipped_count} files (already set or no augmentation section)")


if __name__ == "__main__":
    main()
