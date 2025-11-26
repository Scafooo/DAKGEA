#!/usr/bin/env python3
"""Interactive tuning with transformation visualization.

Shows actual transformations for manual evaluation before running full tuning.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, set_global_level
from src.config.loader import load_yaml

logger = get_logger("interactive_tuning")


def load_dataset_examples(dataset_name: str, ratio: float, n_examples: int = 10) -> List[Tuple[str, str]]:
    """Load example value pairs from the dataset for visualization.

    Args:
        dataset_name: Dataset name (e.g., "BBC_DB")
        ratio: Reduction ratio
        n_examples: Number of examples to load

    Returns:
        List of (value1, value2) pairs
    """
    from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory

    logger.info(f"Loading {n_examples} example pairs from {dataset_name}...")

    # Load dataset
    reader = DatasetReaderFactory.create_reader("openea")
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "openea" / dataset_name

    if not raw_data_path.exists():
        logger.warning(f"Dataset path not found: {raw_data_path}")
        return []

    dataset = reader.read(str(raw_data_path))

    # Get alignment pairs
    alignment = dataset.aligned_pairs
    pairs = list(alignment.items())[:n_examples]

    # Extract literal values from entities
    examples = []
    for src_uri, tgt_uri in pairs:
        src_kg = dataset.source_kg
        tgt_kg = dataset.target_kg

        # Get some literal values from source entity
        src_values = []
        for _, _, obj in src_kg.graph.triples((src_uri, None, None)):
            if isinstance(obj, str) and len(obj) > 3 and len(obj) < 100:
                src_values.append(obj)

        # Get some literal values from target entity
        tgt_values = []
        for _, _, obj in tgt_kg.graph.triples((tgt_uri, None, None)):
            if isinstance(obj, str) and len(obj) > 3 and len(obj) < 100:
                tgt_values.append(obj)

        # Pair up values
        for src_val in src_values[:3]:
            for tgt_val in tgt_values[:3]:
                if src_val != tgt_val:  # Skip identical values
                    examples.append((src_val, tgt_val))
                    if len(examples) >= n_examples:
                        return examples

    return examples


def test_transformation(val1: str, val2: str, config: Dict[str, Any]) -> Tuple[str, str]:
    """Test transformation with given config.

    Args:
        val1: First value
        val2: Second value
        config: Configuration dict

    Returns:
        Tuple of (transformed1, transformed2)
    """
    from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM

    # Extract BART config
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    # Initialize interpolator (force GPU usage)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    interpolator = BartInterpolatorPLM(
        model_name=bart_cfg.get("model_name", "facebook/bart-base"),
        out_dir=bart_cfg.get("out_dir", "./bart_plm_model_base"),
        device=device,
        base_alpha=bart_cfg.get("base_alpha", 0.5),
        alpha_spread=bart_cfg.get("alpha_spread", 0.45),
        max_len_in=bart_cfg.get("max_len_in", 96),
        max_len_out=bart_cfg.get("max_len_out", 48),
        generation_config=gen_cfg,
        advanced_training_config=bart_cfg.get("advanced_training", {}),
    )

    # Perform interpolation
    result1, result2 = interpolator.interpolate_pair(val1, val2)

    return result1, result2


def print_transformation(val1: str, val2: str, result1: str, result2: str, params: Dict[str, Any]):
    """Pretty print a transformation."""
    print("\n" + "="*80)
    print("TRANSFORMATION EXAMPLE")
    print("="*80)

    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*80)
    print("INPUT:")
    print(f"  Source: {val1}")
    print(f"  Target: {val2}")

    print("\nOUTPUT:")
    print(f"  Source: {result1}")
    print(f"  Target: {result2}")

    print("\nCHANGES:")
    if result1 == val1:
        print(f"  Source: ❌ NO CHANGE (identical)")
    else:
        print(f"  Source: ✓ Changed")

    if result2 == val2:
        print(f"  Target: ❌ NO CHANGE (identical)")
    else:
        print(f"  Target: ✓ Changed")

    # Check if outputs are similar to each other
    if result1.lower() == result2.lower():
        print(f"  Similarity: ⚠️  OUTPUTS ARE IDENTICAL")
    elif result1.lower() in result2.lower() or result2.lower() in result1.lower():
        print(f"  Similarity: ⚠️  One output contains the other")
    else:
        print(f"  Similarity: ✓ Outputs are distinct")

    print("="*80)


def interactive_session(dataset: str, ratio: float, config_path: Path):
    """Run interactive tuning session."""
    logger.info("="*80)
    logger.info("INTERACTIVE TUNING SESSION")
    logger.info("="*80)

    # Set verbose logging to see transformations
    set_global_level("INFO")

    # Load base config
    config = load_yaml(config_path)

    # Load example pairs
    examples = load_dataset_examples(dataset, ratio, n_examples=5)

    if not examples:
        logger.error("Could not load examples from dataset")
        return

    logger.info(f"Loaded {len(examples)} example pairs")

    # Get current parameters
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    current_params = {
        "base_alpha": bart_cfg.get("base_alpha", 0.5),
        "alpha_spread": bart_cfg.get("alpha_spread", 0.45),
        "temperature": gen_cfg.get("temperature", 0.85),
        "top_p": gen_cfg.get("top_p", 0.9),
        "num_beams": gen_cfg.get("num_beams", 5),
        "repetition_penalty": gen_cfg.get("repetition_penalty", 1.7),
        "noise_std": gen_cfg.get("noise_std", 0.001),
    }

    print("\n" + "="*80)
    print("CURRENT PARAMETERS")
    print("="*80)
    for key, value in current_params.items():
        print(f"  {key}: {value}")
    print("="*80)

    # Test with current parameters
    print("\n\n🔍 TESTING WITH CURRENT PARAMETERS...")

    for i, (val1, val2) in enumerate(examples[:3], 1):
        print(f"\n📝 Example {i}/{min(3, len(examples))}")
        try:
            result1, result2 = test_transformation(val1, val2, config)
            print_transformation(val1, val2, result1, result2, current_params)
        except Exception as e:
            logger.error(f"Transformation failed: {e}")

    # Interactive loop
    while True:
        print("\n" + "="*80)
        print("OPTIONS")
        print("="*80)
        print("1. Change parameter and test")
        print("2. Test with different examples")
        print("3. Save current config")
        print("4. Run full tuning with these params")
        print("5. Exit")
        print("="*80)

        choice = input("\nChoice (1-5): ").strip()

        if choice == "1":
            # Change parameter
            print("\nAvailable parameters:")
            for i, key in enumerate(current_params.keys(), 1):
                print(f"  {i}. {key} = {current_params[key]}")

            param_choice = input("\nParameter number to change: ").strip()
            try:
                param_idx = int(param_choice) - 1
                param_name = list(current_params.keys())[param_idx]
                new_value = input(f"New value for {param_name}: ").strip()

                # Parse value
                try:
                    if "." in new_value:
                        new_value = float(new_value)
                    else:
                        new_value = int(new_value)
                except:
                    pass

                current_params[param_name] = new_value

                # Update config
                if param_name in ["base_alpha", "alpha_spread"]:
                    bart_cfg[param_name] = new_value
                else:
                    gen_cfg[param_name] = new_value

                # Test with new parameter
                print(f"\n🔍 TESTING WITH {param_name}={new_value}...")

                for i, (val1, val2) in enumerate(examples[:2], 1):
                    print(f"\n📝 Example {i}/2")
                    try:
                        result1, result2 = test_transformation(val1, val2, config)
                        print_transformation(val1, val2, result1, result2, current_params)
                    except Exception as e:
                        logger.error(f"Transformation failed: {e}")

            except Exception as e:
                logger.error(f"Invalid input: {e}")

        elif choice == "2":
            # Test with different examples
            print("\n🔍 TESTING WITH DIFFERENT EXAMPLES...")
            examples = load_dataset_examples(dataset, ratio, n_examples=5)

            for i, (val1, val2) in enumerate(examples[:3], 1):
                print(f"\n📝 Example {i}/3")
                try:
                    result1, result2 = test_transformation(val1, val2, config)
                    print_transformation(val1, val2, result1, result2, current_params)
                except Exception as e:
                    logger.error(f"Transformation failed: {e}")

        elif choice == "3":
            # Save config
            output_path = PROJECT_ROOT / "config" / "augmentation" / "plm_tuned.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(config, f)
            logger.info(f"✓ Config saved to {output_path}")

        elif choice == "4":
            # Run full tuning
            print("\n⚠️  Starting full tuning with current parameters...")
            print("This will take several minutes. Continue? (y/n)")
            if input().strip().lower() == 'y':
                from experiments.quick_test import main as quick_test_main
                # TODO: implement full run
                logger.info("Running full augmentation test...")
                break

        elif choice == "5":
            logger.info("Exiting interactive session")
            break

        else:
            print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(description="Interactive tuning with visualization")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--ratio", type=float, required=True, help="Reduction ratio")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "augmentation" / "plm.yaml", help="Config file")

    args = parser.parse_args()

    interactive_session(args.dataset, args.ratio, args.config)


if __name__ == "__main__":
    main()
