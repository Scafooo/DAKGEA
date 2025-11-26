#!/usr/bin/env python3
"""Test multiple parameter configurations to find optimal settings.

This script tests different parameter combinations and compares their
transformation quality to identify the best settings.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, set_global_level
from src.config.loader import load_yaml

set_global_level("WARNING")
logger = get_logger(__name__)


# Test configurations to try
PARAMETER_CONFIGS = {
    "current": {
        "name": "Current Config",
        "temperature": 0.85,
        "top_p": 0.9,
        "num_beams": 5,
        "repetition_penalty": 1.7,
        "noise_std": 0.001,
    },
    "high_creativity": {
        "name": "High Creativity",
        "temperature": 1.1,
        "top_p": 0.92,
        "num_beams": 3,
        "repetition_penalty": 1.5,
        "noise_std": 0.01,
    },
    "low_beams_high_temp": {
        "name": "Low Beams + High Temp",
        "temperature": 1.0,
        "top_p": 0.9,
        "num_beams": 3,
        "repetition_penalty": 1.7,
        "noise_std": 0.005,
    },
    "high_noise": {
        "name": "High Noise",
        "temperature": 0.9,
        "top_p": 0.9,
        "num_beams": 5,
        "repetition_penalty": 1.7,
        "noise_std": 0.05,
    },
    "balanced": {
        "name": "Balanced",
        "temperature": 0.95,
        "top_p": 0.92,
        "num_beams": 4,
        "repetition_penalty": 1.6,
        "noise_std": 0.01,
    },
}


def test_config(config_name: str, params: Dict, base_config: Dict, test_pairs: List[Tuple[str, str]]):
    """Test a specific parameter configuration."""
    import torch
    from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM

    # Create modified config
    test_config = copy.deepcopy(base_config)
    gen_cfg = test_config["augmentation"]["bart"]["generation"]

    # Update generation parameters
    for key, value in params.items():
        if key == "noise_std":
            gen_cfg[key] = value
        else:
            gen_cfg[key] = value

    bart_cfg = test_config["augmentation"]["bart"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*80}")
    print(f"TESTING: {params['name']}")
    print("="*80)
    print("Parameters:")
    for key, value in params.items():
        if key != "name":
            print(f"  {key:25s} = {value}")
    print()

    # Initialize interpolator
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

    # Test transformations
    swap_count = 0
    copy_count = 0
    interp_count = 0

    for i, (val1, val2) in enumerate(test_pairs, 1):
        result1, result2 = interpolator.interpolate_pair(val1, val2)

        # Analyze pattern
        is_swap = result1.lower() == val2.lower() and result2.lower() == val1.lower()
        is_copy = result1 == val1 and result2 == val2
        is_interp = result1 != val1 and result2 != val2 and not is_swap

        if is_swap:
            swap_count += 1
            status = "SWAP"
        elif is_copy:
            copy_count += 1
            status = "COPY"
        elif is_interp:
            interp_count += 1
            status = "INTERP"
        else:
            status = "OTHER"

        print(f"{i:2d}. [{status:6s}] '{val1}' / '{val2}' → '{result1}' / '{result2}'")

    n_tests = len(test_pairs)
    swap_pct = swap_count / n_tests * 100
    copy_pct = copy_count / n_tests * 100
    interp_pct = interp_count / n_tests * 100

    print(f"\nResults:")
    print(f"  Swapping:      {swap_count:2d} ({swap_pct:5.1f}%)")
    print(f"  Copying:       {copy_count:2d} ({copy_pct:5.1f}%)")
    print(f"  Interpolation: {interp_count:2d} ({interp_pct:5.1f}%)")

    # Score (interpolation is good, swapping/copying is bad)
    score = interp_pct - (swap_pct + copy_pct) * 0.5

    print(f"\n  Quality Score: {score:6.2f}  (higher = better)")

    return {
        "config_name": config_name,
        "params": params,
        "swap_count": swap_count,
        "copy_count": copy_count,
        "interp_count": interp_count,
        "swap_pct": swap_pct,
        "copy_pct": copy_pct,
        "interp_pct": interp_pct,
        "score": score,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test parameter configurations")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "augmentation" / "plm.yaml",
                        help="Base config file")
    parser.add_argument("--configs", nargs="+",
                        choices=list(PARAMETER_CONFIGS.keys()) + ["all"],
                        default=["current"],
                        help="Which configs to test")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        return 1

    # Load base config
    base_config = load_yaml(args.config)

    # Test pairs
    test_pairs = [
        ("BBC", "British Broadcasting Corporation"),
        ("New York", "New York City"),
        ("England", "England, UK"),
        ("France", "Germany"),
        ("rock music", "classical music"),
        ("01t00 1969 04 00 01", "01t00 1975 04 00 01"),
        ("singer", "vocalist"),
        ("London", "Paris"),
    ]

    # Determine which configs to test
    if "all" in args.configs:
        configs_to_test = list(PARAMETER_CONFIGS.keys())
    else:
        configs_to_test = args.configs

    print("="*80)
    print("PARAMETER CONFIGURATION COMPARISON")
    print("="*80)
    print(f"Testing {len(configs_to_test)} configurations with {len(test_pairs)} test pairs\n")

    # Test all configurations
    results = []
    for config_name in configs_to_test:
        if config_name not in PARAMETER_CONFIGS:
            print(f"⚠️  Unknown config: {config_name}, skipping")
            continue

        params = PARAMETER_CONFIGS[config_name]
        result = test_config(config_name, params, base_config, test_pairs)
        results.append(result)

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Config':<20s} {'Swap%':>8s} {'Copy%':>8s} {'Interp%':>8s} {'Score':>8s}")
    print("-"*80)

    # Sort by score (best first)
    results.sort(key=lambda x: x["score"], reverse=True)

    for result in results:
        name = result["params"]["name"]
        print(f"{name:<20s} {result['swap_pct']:>7.1f}% {result['copy_pct']:>7.1f}% "
              f"{result['interp_pct']:>7.1f}% {result['score']:>7.2f}")

    # Best config
    if results:
        best = results[0]
        print(f"\n{'='*80}")
        print(f"🏆 BEST CONFIGURATION: {best['params']['name']}")
        print("="*80)
        print("Parameters to use:")
        for key, value in best["params"].items():
            if key != "name":
                print(f"  {key}: {value}")
        print("\nTo apply these settings, update config/augmentation/plm.yaml:")
        print("```yaml")
        print("  generation:")
        for key, value in best["params"].items():
            if key != "name":
                print(f"    {key}: {value}")
        print("```")

    return 0


if __name__ == "__main__":
    sys.exit(main())
