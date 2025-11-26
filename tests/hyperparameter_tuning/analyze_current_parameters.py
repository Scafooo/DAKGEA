#!/usr/bin/env python3
"""Analyze current parameter configuration and transformation quality.

This script helps identify which parameters need tuning by:
1. Testing transformations with current config
2. Analyzing output patterns (swapping, copying, interpolation)
3. Suggesting parameter adjustments
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, set_global_level
from src.config.loader import load_yaml

set_global_level("WARNING")
logger = get_logger(__name__)


def analyze_transformation(val1: str, val2: str, result1: str, result2: str) -> dict:
    """Analyze a transformation to categorize its behavior."""
    analysis = {
        "is_swap": False,
        "is_copy": False,
        "is_interpolation": False,
        "src_changed": result1 != val1,
        "tgt_changed": result2 != val2,
        "outputs_identical": result1.lower() == result2.lower(),
        "length_ratio_src": len(result1) / max(len(val1), 1),
        "length_ratio_tgt": len(result2) / max(len(val2), 1),
    }

    # Check for simple swapping
    if result1.lower() == val2.lower() and result2.lower() == val1.lower():
        analysis["is_swap"] = True

    # Check for copying (no change)
    elif not analysis["src_changed"] and not analysis["tgt_changed"]:
        analysis["is_copy"] = True

    # Check for interpolation (both changed, distinct outputs)
    elif analysis["src_changed"] and analysis["tgt_changed"] and not analysis["outputs_identical"]:
        analysis["is_interpolation"] = True

    return analysis


def test_transformations_with_config(config_path: Path, n_tests: int = 10):
    """Test transformations with given config and analyze patterns."""
    import torch
    from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM

    config = load_yaml(config_path)
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Test examples with varying similarity
    test_pairs = [
        ("BBC", "British Broadcasting Corporation"),  # Acronym expansion
        ("New York", "New York City"),  # Partial overlap
        ("England", "England, UK"),  # Same with addition
        ("France", "Germany"),  # Different but same type
        ("rock music", "classical music"),  # Same structure
        ("01t00 1969 04 00 01", "01t00 1975 04 00 01"),  # Dates
        ("singer", "vocalist"),  # Synonyms
        ("London", "Paris"),  # Different cities
        ("jazz", "blues"),  # Music genres
        ("actor", "actress"),  # Gender variants
    ]

    print("="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    print(f"  base_alpha:           {bart_cfg.get('base_alpha')}")
    print(f"  alpha_spread:         {bart_cfg.get('alpha_spread')}")
    print(f"  temperature:          {gen_cfg.get('temperature')}")
    print(f"  top_p:                {gen_cfg.get('top_p')}")
    print(f"  num_beams:            {gen_cfg.get('num_beams')}")
    print(f"  repetition_penalty:   {gen_cfg.get('repetition_penalty')}")
    print(f"  noise_std:            {gen_cfg.get('noise_std')}")
    print(f"  enable_retry:         {bart_cfg.get('enable_retry_on_identical_tokens', True)}")
    print(f"  retry_threshold:      {gen_cfg.get('identical_tokens_threshold')}")
    print("="*80 + "\n")

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

    # Run transformations and analyze
    results = []
    patterns = {
        "swap": 0,
        "copy": 0,
        "interpolation": 0,
        "other": 0,
    }

    for i, (val1, val2) in enumerate(test_pairs[:n_tests], 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{n_tests}")
        print(f"{'='*80}")
        print(f"Input:  '{val1}' / '{val2}'")

        result1, result2 = interpolator.interpolate_pair(val1, val2)

        print(f"Output: '{result1}' / '{result2}'")

        analysis = analyze_transformation(val1, val2, result1, result2)

        # Categorize
        if analysis["is_swap"]:
            category = "SWAP"
            patterns["swap"] += 1
            print(f"❌ Pattern: SWAP (simply swapped inputs)")
        elif analysis["is_copy"]:
            category = "COPY"
            patterns["copy"] += 1
            print(f"❌ Pattern: COPY (no transformation)")
        elif analysis["is_interpolation"]:
            category = "INTERPOLATION"
            patterns["interpolation"] += 1
            print(f"✅ Pattern: INTERPOLATION (proper transformation)")
        else:
            category = "OTHER"
            patterns["other"] += 1
            print(f"⚠️  Pattern: OTHER (partial or unusual transformation)")

        results.append({
            "input": (val1, val2),
            "output": (result1, result2),
            "analysis": analysis,
            "category": category,
        })

    # Summary
    print(f"\n\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total tests: {n_tests}")
    print(f"  Swapping:       {patterns['swap']:2d} ({patterns['swap']/n_tests*100:5.1f}%)")
    print(f"  Copying:        {patterns['copy']:2d} ({patterns['copy']/n_tests*100:5.1f}%)")
    print(f"  Interpolation:  {patterns['interpolation']:2d} ({patterns['interpolation']/n_tests*100:5.1f}%)")
    print(f"  Other:          {patterns['other']:2d} ({patterns['other']/n_tests*100:5.1f}%)")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)

    swap_rate = patterns['swap'] / n_tests
    copy_rate = patterns['copy'] / n_tests
    interp_rate = patterns['interpolation'] / n_tests

    if swap_rate > 0.5:
        print("⚠️  HIGH SWAPPING RATE (>50%)")
        print("   Problem: Model is just swapping inputs instead of interpolating")
        print("   Suggestions:")
        print("   - Increase temperature (current: {}) → try 1.0-1.2".format(gen_cfg.get('temperature')))
        print("   - Increase noise_std (current: {}) → try 0.01-0.05".format(gen_cfg.get('noise_std')))
        print("   - Decrease num_beams (current: {}) → try 3 (more randomness)".format(gen_cfg.get('num_beams')))
        print("   - Ensure BART fine-tuning is working correctly")

    if copy_rate > 0.3:
        print("\n⚠️  HIGH COPYING RATE (>30%)")
        print("   Problem: Model is copying inputs unchanged")
        print("   Suggestions:")
        print("   - Enable retry mechanism (enable_retry_on_identical_tokens)")
        print("   - Lower identical_tokens_threshold → try 0.2")
        print("   - Increase noise injection")

    if interp_rate < 0.3:
        print("\n⚠️  LOW INTERPOLATION RATE (<30%)")
        print("   Problem: Model rarely produces proper interpolations")
        print("   Suggestions:")
        print("   - Check if BART model is properly fine-tuned")
        print("   - Verify training data quality")
        print("   - Try different alpha_spread values")

    if interp_rate >= 0.6:
        print("\n✅ GOOD INTERPOLATION RATE (≥60%)")
        print("   Current parameters are working reasonably well")
        print("   Consider fine-tuning for marginal improvements")

    print("="*80 + "\n")

    return results, patterns


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze current PLM parameters")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "augmentation" / "plm.yaml",
                        help="Config file to analyze")
    parser.add_argument("--tests", type=int, default=10,
                        help="Number of test transformations")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        return 1

    test_transformations_with_config(args.config, args.tests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
