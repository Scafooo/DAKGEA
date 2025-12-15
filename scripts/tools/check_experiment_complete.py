#!/usr/bin/env python3
"""
Quick pre-check to determine if an experiment is already complete.
Used by parallel runner to skip already-completed experiments without loading datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path
import yaml


def check_experiment_complete(config_path: Path, results_base: Path) -> bool:
    """
    Fast check if experiment is complete without loading any datasets or models.

    Returns True if all required result files exist.
    """
    # Load YAML config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name")
    if not exp_name:
        return False

    # Check overwrite_existing flag
    overwrite = exp_cfg.get("overwrite_existing", False)
    if overwrite:
        # Will overwrite, so not complete
        return False

    # Get eval flags
    reduction_cfg = exp_cfg.get("reduction", {})
    augmentation_cfg = exp_cfg.get("augmentation", {})
    reduction_eval = reduction_cfg.get("eval", False)
    augmentation_eval = augmentation_cfg.get("eval", False)

    # Get augmentation method
    aug_method = augmentation_cfg.get("method")
    if aug_method == "stub":
        aug_method = None

    # Get models
    model = exp_cfg.get("model")
    models = [model] if model else []

    # Workspace path (support suite grouping)
    suite = exp_cfg.get("suite")
    if suite:
        workspace = results_base / suite / exp_name
    else:
        workspace = results_base / exp_name
    if not workspace.exists():
        return False

    # Find the ratio directory (should be only one for these experiments)
    ratio_dirs = list(workspace.glob("ratio_*"))
    if not ratio_dirs:
        return False

    # Check each ratio directory
    for ratio_dir in ratio_dirs:
        dataset_dirs = list(ratio_dir.glob("*"))
        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                continue

            artifact_root = dataset_dir / "artifacts"
            if not artifact_root.exists():
                return False

            # Check reduction results if required
            if reduction_eval:
                reduction_results = dataset_dir / "reduction" / "results.json"
                if not reduction_results.exists():
                    return False

            # Check augmentation results if required
            if aug_method and augmentation_eval:
                augmentation_results = dataset_dir / "augmentation" / "results.json"
                if not augmentation_results.exists():
                    return False

            # Check evaluation results for all models
            evaluation_root = artifact_root / "evaluation"
            for model_name in models:
                # Check baseline evaluation
                if reduction_eval:
                    baseline_result = evaluation_root / "baseline" / f"{model_name}.json"
                    if not baseline_result.exists():
                        return False

                # Check augmentation evaluation
                if aug_method and augmentation_eval:
                    aug_result = evaluation_root / aug_method / f"{model_name}.json"
                    if not aug_result.exists():
                        return False

    return True


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: check_experiment_complete.py CONFIG_FILE", file=sys.stderr)
        sys.exit(2)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    # Get results base path from project root
    project_root = Path(__file__).parent.parent
    results_base = project_root / "results"

    try:
        is_complete = check_experiment_complete(config_path, results_base)
        if is_complete:
            # Exit code 0 = complete
            sys.exit(0)
        else:
            # Exit code 1 = not complete
            sys.exit(1)
    except Exception as e:
        # Exit code 2 = error checking
        print(f"Error checking experiment: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
