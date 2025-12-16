#!/usr/bin/env python3
"""
Script to update suite names in multilingual config files
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def update_suite_name(config_path: Path, new_suite_name: str):
    """Update the suite name in a config file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["experiment"]["suite"] = new_suite_name

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    massive_dir = PROJECT_ROOT / "config" / "experiments" / "massive"

    # Map of directory to suite name pattern
    multilingual_dirs = {
        "multilingual_bert_int_baseline": "massive_multilingual_baseline_bert_int",
        "multilingual_bert_int_aug_red": "massive_multilingual_aug_red_bert_int",
        "multilingual_bert_int_synthetic_only": "massive_multilingual_synthetic_only_bert_int",
        "multilingual_rrea_baseline": "massive_multilingual_baseline_rrea",
        "multilingual_rrea_aug_red": "massive_multilingual_aug_red_rrea",
        "multilingual_rrea_synthetic_only": "massive_multilingual_synthetic_only_rrea",
    }

    total_updated = 0

    for dir_name, suite_name in multilingual_dirs.items():
        target_dir = massive_dir / dir_name
        if not target_dir.exists():
            print(f"Directory {dir_name} does not exist, skipping")
            continue

        config_files = list(target_dir.glob("*.yaml"))
        print(f"Updating {len(config_files)} files in {dir_name}...")

        for config_file in config_files:
            update_suite_name(config_file, suite_name)
            total_updated += 1

    print(f"\nCompleted! Updated {total_updated} config files")


if __name__ == "__main__":
    main()
