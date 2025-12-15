"""Example: How to use training_mode for synthetic vs real comparison.

The training_mode is now integrated directly into PLMAugmenter.
No need for manual filtering - just set the mode in config.
"""

from src.core.dataset import Dataset
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter


def example_training_pipeline(config: dict):
    """Example of training pipeline with different modes.

    Args:
        config: Configuration dictionary with augmentation.training_mode
    """
    # 1. Load original dataset
    print("[1] Loading dataset...")
    dataset_original = Dataset.load(config["dataset_name"])
    print(f"    Original aligned pairs: {len(dataset_original.aligned_entities)}")

    # 2. Apply augmentation (with automatic training_mode filtering)
    print("\n[2] Applying augmentation...")
    augmenter = PLMAugmenter(config)
    dataset_final = augmenter.augment(dataset_original)

    # The augmenter automatically applies training_mode filtering:
    # - "baseline": Returns only original pairs (no augmentation)
    # - "synthetic_only": Returns only synthetic pairs (removes originals)
    # - "augmented": Returns all pairs (original + synthetic) [default]

    print(f"    Final training pairs: {len(dataset_final.aligned_entities)}")

    # 3. Train model (placeholder)
    print("\n[3] Training model...")
    print(f"    Training with {len(dataset_final.aligned_entities)} pairs")
    print(f"    Mode: {config.get('augmentation', {}).get('training_mode', 'augmented')}")

    # train_model(dataset_final, config["training"])

    return dataset_final


def run_all_modes_comparison():
    """Run comparison across all three modes."""
    config_base = {
        "dataset_name": "D_W_15K_V1",
        "augmentation": {
            "max_depth": 1,
            "ratio": 1.0,
            "bart": {"enable_finetuning": True},
            # training_mode will be set per iteration
        },
        "training": {
            "epochs": 100,
            "batch_size": 256
        }
    }

    modes = ["baseline", "synthetic_only", "augmented"]
    results = {}

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"RUNNING MODE: {mode}")
        print("=" * 80)

        # Create config with specific training_mode
        config = {
            **config_base,
            "augmentation": {
                **config_base["augmentation"],
                "training_mode": mode,  # ← Set mode here
            }
        }

        dataset = example_training_pipeline(config)
        results[mode] = {
            "num_pairs": len(dataset.aligned_entities),
            # Add actual training results here
        }

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for mode, result in results.items():
        print(f"{mode:>15}: {result['num_pairs']:>6} training pairs")


if __name__ == "__main__":
    # Example 1: Synthetic-only mode
    print("Example 1: Running synthetic_only mode")
    print("=" * 80)

    config_synthetic = {
        "dataset_name": "D_W_15K_V1",
        "augmentation": {
            "max_depth": 1,
            "ratio": 1.0,
            "training_mode": "synthetic_only",  # ← KEY: Use ONLY synthetic data
        },
        "training": {
            "epochs": 100
        }
    }

    # example_training_pipeline(config_synthetic)

    # Example 2: Baseline mode (no augmentation)
    print("\n\nExample 2: Running baseline mode")
    print("=" * 80)

    config_baseline = {
        "dataset_name": "D_W_15K_V1",
        "augmentation": {
            "max_depth": 1,
            "ratio": 1.0,
            "training_mode": "baseline",  # ← KEY: Use ONLY original data
        },
        "training": {
            "epochs": 100
        }
    }

    # example_training_pipeline(config_baseline)

    # Example 3: All modes comparison
    print("\n\nExample 3: Running all modes for comparison")
    print("=" * 80)
    # run_all_modes_comparison()

    print("\n✓ Examples completed! (Uncomment function calls to actually run)")
    print("\nKey takeaway:")
    print("  - Just set augmentation.training_mode in your config")
    print("  - PLMAugmenter handles filtering automatically")
    print("  - No manual filter_dataset_by_mode() needed!")
