#!/usr/bin/env python3
"""Interactive testing with hardcoded dataset-specific examples.

Shows transformations iteratively, allowing parameter adjustments between tests.
Focuses on real attribute values from each dataset.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, set_global_level
from src.config.loader import load_yaml
import yaml

set_global_level("WARNING")
logger = get_logger(__name__)


# Hardcoded test examples from real datasets
DATASET_EXAMPLES = {
    "BBC_DB": [
        # From music artists - names vs dates
        ("braxtons", "01t00 04 1989 00 01"),
        ("braxtons", "band or group"),
        ("bob geldof", "01t00 1975 04 00 01"),
        ("faith blind", "01t00 1969 04 00 01"),

        # Artist names
        ("the beatles", "the rolling stones"),
        ("led zeppelin", "pink floyd"),

        # Genres
        ("rock music", "classical music"),
        ("jazz", "blues"),

        # Locations
        ("london", "liverpool"),
        ("new york", "los angeles"),
    ],

    "D_W_15K_V1": [
        # From DBpedia-Wikidata - mixed attributes
        ('"Beautiful, Loved and Blessed"', 'black-sweat-lyrics-prince'),
        ("United States", "USA"),
        ("English", "en"),

        # Dates and numbers
        ("1990-01-15", "1995-06-22"),
        ("2000000", "3500000"),

        # Names and titles
        ("John Smith", "Jane Doe"),
        ("The Matrix", "Inception"),
    ],

    "ICEW_WIKI": [
        # Events and dates
        ("2019-05-12", "2020-03-18"),
        ("diplomatic visit", "state meeting"),

        # Locations
        ("Washington D.C.", "Beijing"),
        ("European Union", "United Nations"),

        # Political terms
        ("president", "prime minister"),
        ("agreement", "treaty"),
    ],

    "ICEW_YAGO": [
        # Similar to ICEW_WIKI
        ("military operation", "peacekeeping mission"),
        ("2018-07-01", "2019-12-25"),

        # Entities
        ("NATO", "UN"),
        ("United Kingdom", "France"),
    ],
}


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def analyze_transformation(val1: str, val2: str, result1: str, result2: str) -> dict:
    """Analyze transformation pattern."""
    return {
        "is_swap": result1.lower().strip() == val2.lower().strip() and
                   result2.lower().strip() == val1.lower().strip(),
        "is_copy": result1 == val1 and result2 == val2,
        "src_changed": result1 != val1,
        "tgt_changed": result2 != val2,
        "outputs_identical": result1.lower().strip() == result2.lower().strip(),
    }


def check_encoding_issues(text: str) -> List[str]:
    """Check for potential encoding issues in text."""
    issues = []

    # Check for common encoding problems
    if "\\u" in text or "\\x" in text:
        issues.append(f"Escape sequences found: {text}")

    # Check for non-ASCII characters that might not render
    try:
        text.encode('ascii')
    except UnicodeEncodeError as e:
        issues.append(f"Non-ASCII characters: {e}")

    # Check for replacement characters
    if "\ufffd" in text or "�" in text:
        issues.append("Replacement character (�) found - encoding corruption")

    return issues


def test_examples_interactive(dataset_name: str, examples: List[Tuple[str, str]],
                              config_path: Path, current_params: Dict):
    """Test examples interactively with current parameters."""
    import torch
    from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM

    # Load and update config
    config = load_yaml(config_path)
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    # Apply current parameters
    for key, value in current_params.items():
        if key in ["base_alpha", "alpha_spread"]:
            bart_cfg[key] = value
        else:
            gen_cfg[key] = value

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_separator()
    print(f"TESTING DATASET: {dataset_name}")
    print_separator()
    print(f"Device: {device}")
    print(f"Examples: {len(examples)}")
    print()
    print("Current Parameters:")
    for key, value in sorted(current_params.items()):
        print(f"  {key:25s} = {value}")
    print()

    print("Loading BART model...")
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
    print("✓ Model loaded\n")

    # Statistics
    patterns = {"swap": 0, "copy": 0, "interpolation": 0, "other": 0}
    encoding_issues_found = []

    for i, (val1, val2) in enumerate(examples, 1):
        print_separator("-")
        print(f"Test {i}/{len(examples)}")
        print_separator("-")

        # Check encoding issues in inputs
        input_encoding_issues = []
        input_encoding_issues.extend(check_encoding_issues(val1))
        input_encoding_issues.extend(check_encoding_issues(val2))

        if input_encoding_issues:
            print("⚠️  INPUT ENCODING ISSUES:")
            for issue in input_encoding_issues:
                print(f"    {issue}")

        print(f"Input:  '{val1}' / '{val2}'")

        # Perform transformation
        result1, result2 = interpolator.interpolate_pair(val1, val2)

        print(f"Output: '{result1}' / '{result2}'")

        # Check encoding issues in outputs
        output_encoding_issues = []
        output_encoding_issues.extend(check_encoding_issues(result1))
        output_encoding_issues.extend(check_encoding_issues(result2))

        if output_encoding_issues:
            print("⚠️  OUTPUT ENCODING ISSUES:")
            for issue in output_encoding_issues:
                print(f"    {issue}")
            encoding_issues_found.append((i, val1, val2, result1, result2, output_encoding_issues))

        # Analyze pattern
        analysis = analyze_transformation(val1, val2, result1, result2)

        if analysis["is_swap"]:
            print("❌ Pattern: SWAP (simply swapped inputs)")
            patterns["swap"] += 1
        elif analysis["is_copy"]:
            print("❌ Pattern: COPY (no transformation)")
            patterns["copy"] += 1
        elif analysis["src_changed"] and analysis["tgt_changed"] and not analysis["outputs_identical"]:
            print("✅ Pattern: INTERPOLATION (proper transformation)")
            patterns["interpolation"] += 1
        else:
            print("⚠️  Pattern: OTHER")
            patterns["other"] += 1

        print()

    # Summary
    print_separator()
    print("SUMMARY")
    print_separator()
    n_tests = len(examples)
    for pattern_name, count in patterns.items():
        pct = count / n_tests * 100
        print(f"{pattern_name.capitalize():15s}: {count:2d} ({pct:5.1f}%)")

    score = patterns["interpolation"] / n_tests * 100 - (patterns["swap"] + patterns["copy"]) / n_tests * 50
    print(f"\nQuality Score: {score:6.2f} (higher = better)")

    # Encoding issues summary
    if encoding_issues_found:
        print(f"\n⚠️  ENCODING ISSUES DETECTED: {len(encoding_issues_found)} cases")
        print("These may indicate:")
        print("  - Unicode characters not properly decoded")
        print("  - Character encoding mismatch (UTF-8 vs Latin-1)")
        print("  - Data corruption in dataset files")

    return {
        "dataset": dataset_name,
        "patterns": patterns,
        "score": score,
        "encoding_issues": encoding_issues_found,
    }


def interactive_session(config_path: Path):
    """Run interactive hyperparameter tuning session."""

    # Starting parameters (from config)
    config = load_yaml(config_path)
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    current_params = {
        "temperature": gen_cfg.get("temperature", 0.85),
        "top_p": gen_cfg.get("top_p", 0.9),
        "num_beams": gen_cfg.get("num_beams", 5),
        "repetition_penalty": gen_cfg.get("repetition_penalty", 1.7),
        "noise_std": gen_cfg.get("noise_std", 0.001),
        "base_alpha": bart_cfg.get("base_alpha", 0.5),
        "alpha_spread": bart_cfg.get("alpha_spread", 0.45),
    }

    print("\n" + "="*80)
    print("INTERACTIVE DATASET TESTING")
    print("="*80)
    print("\nThis tool tests hardcoded examples from each dataset")
    print("You can adjust parameters iteratively to find optimal settings\n")

    while True:
        print("\n" + "="*80)
        print("MENU")
        print("="*80)
        print("1. Test all datasets with current parameters")
        print("2. Test specific dataset")
        print("3. Adjust parameters")
        print("4. Save current parameters to config")
        print("5. Quit")
        print("="*80)

        choice = input("\nChoice (1-5): ").strip()

        if choice == "1":
            # Test all datasets
            results = []
            for dataset_name, examples in DATASET_EXAMPLES.items():
                result = test_examples_interactive(dataset_name, examples, config_path, current_params)
                results.append(result)
                input("\nPress Enter to continue to next dataset...")

            # Overall summary
            print("\n" + "="*80)
            print("OVERALL SUMMARY")
            print("="*80)
            for result in results:
                print(f"{result['dataset']:15s} - Score: {result['score']:6.2f}")

            avg_score = sum(r["score"] for r in results) / len(results)
            print(f"\nAverage Score: {avg_score:6.2f}")

        elif choice == "2":
            # Test specific dataset
            print("\nAvailable datasets:")
            for i, dataset_name in enumerate(DATASET_EXAMPLES.keys(), 1):
                print(f"  {i}. {dataset_name}")

            dataset_choice = input("\nDataset number: ").strip()
            try:
                dataset_idx = int(dataset_choice) - 1
                dataset_name = list(DATASET_EXAMPLES.keys())[dataset_idx]
                examples = DATASET_EXAMPLES[dataset_name]
                test_examples_interactive(dataset_name, examples, config_path, current_params)
            except (ValueError, IndexError):
                print("❌ Invalid dataset choice")

        elif choice == "3":
            # Adjust parameters
            print("\nCurrent parameters:")
            for i, (key, value) in enumerate(current_params.items(), 1):
                print(f"  {i}. {key:25s} = {value}")

            param_choice = input("\nParameter number to change (or 'back'): ").strip()
            if param_choice.lower() == "back":
                continue

            try:
                param_idx = int(param_choice) - 1
                param_name = list(current_params.keys())[param_idx]

                print(f"\nChanging {param_name} (current: {current_params[param_name]})")
                print("Suggested ranges:")
                suggestions = {
                    "temperature": "0.7-1.2",
                    "top_p": "0.85-0.95",
                    "num_beams": "3-7",
                    "repetition_penalty": "1.3-2.0",
                    "noise_std": "0.0-0.1",
                    "base_alpha": "0.3-0.7",
                    "alpha_spread": "0.2-0.5",
                }
                print(f"  {suggestions.get(param_name, 'varies')}")

                new_value = input(f"New value: ").strip()
                try:
                    if "." in new_value:
                        new_value = float(new_value)
                    else:
                        new_value = int(new_value)

                    current_params[param_name] = new_value
                    print(f"✓ Updated {param_name} = {new_value}")
                except ValueError:
                    print("❌ Invalid value")
            except (ValueError, IndexError):
                print("❌ Invalid parameter choice")

        elif choice == "4":
            # Save parameters
            output_path = PROJECT_ROOT / "config" / "augmentation" / "plm_tuned.yaml"

            # Load base config and update
            config = load_yaml(config_path)
            bart_cfg = config["augmentation"]["bart"]
            gen_cfg = bart_cfg.get("generation", {})

            for key, value in current_params.items():
                if key in ["base_alpha", "alpha_spread"]:
                    bart_cfg[key] = value
                else:
                    gen_cfg[key] = value

            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"✓ Parameters saved to {output_path}")

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("❌ Invalid choice")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive dataset testing")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "augmentation" / "plm.yaml",
                        help="Base config file")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        return 1

    interactive_session(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
