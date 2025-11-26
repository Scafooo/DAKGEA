#!/usr/bin/env python3
"""Visualize augmentation transformations for each dataset separately."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rdflib import Literal
from src.logger import get_logger, set_global_level
from src.config.loader import load_yaml
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory

# Set INFO level to see transformation details
set_global_level("INFO")
logger = get_logger(__name__)


def load_value_pairs_from_dataset(dataset_name: str, n_examples: int = 5):
    """Load diverse value pairs from a dataset using aligned predicates.

    Args:
        dataset_name: Dataset name (e.g., "BBC_DB", "D_W", "ICEWS_WIKI", "ICEWS_YAGO")
        n_examples: Number of example pairs to extract

    Returns:
        List of (value1, value2) tuples from aligned entities
    """
    print(f"\n{'='*80}")
    print(f"Loading examples from {dataset_name}")
    print('='*80)

    reader = DatasetReaderFactory.create_reader("openea")
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "openea" / dataset_name

    if not raw_data_path.exists():
        print(f"  ⚠️  Dataset path not found: {raw_data_path}")
        return []

    dataset = reader.read(str(raw_data_path))
    aligned_entities = dataset.aligned_entities

    # Ground-truth attribute matches (src_pred -> [tgt_pred, ...])
    attr_matches = dataset.attribute_matches if hasattr(dataset, "attribute_matches") else {}

    def _is_human_literal(val: str) -> bool:
        """Heuristic to keep readable values: has letters, bounded length, not mostly digits/codes."""
        val = val.strip()
        if not (3 <= len(val) <= 80):
            return False
        if not any(c.isalpha() for c in val):
            return False
        digits = sum(ch.isdigit() for ch in val)
        if digits / max(1, len(val)) > 0.5:
            return False
        if "t00" in val.lower() or val.lower().startswith("01t"):
            return False
        return True

    def _is_label_pred(pred_uri: str) -> bool:
        name = pred_uri.lower()
        keywords = ("label", "name", "title", "fullname", "altlabel", "nickname", "sortlabel")
        return any(k in name for k in keywords)

    examples = []
    seen_pairs = set()

    for src_uri, tgt_uri in list(aligned_entities):
        if len(examples) >= n_examples:
            break

        src_kg = dataset.knowledge_graph_source
        tgt_kg = dataset.knowledge_graph_target

        src_literals = {}
        for _, pred, obj in src_kg.triples((src_uri, None, None)):
            if isinstance(obj, Literal):
                val = str(obj).strip()
                if _is_label_pred(str(pred)) and _is_human_literal(val):
                    src_literals.setdefault(str(pred), []).append(val)

        tgt_literals = {}
        for _, pred, obj in tgt_kg.triples((tgt_uri, None, None)):
            if isinstance(obj, Literal):
                val = str(obj).strip()
                if _is_label_pred(str(pred)) and _is_human_literal(val):
                    tgt_literals.setdefault(str(pred), []).append(val)

        # Prefer ground-truth predicate matches
        matched = False
        for src_pred, tgt_candidates in attr_matches.items():
            if src_pred not in src_literals:
                continue
            for tgt_pred in tgt_candidates:
                if tgt_pred not in tgt_literals:
                    continue
                for sv in src_literals[src_pred]:
                    for tv in tgt_literals[tgt_pred]:
                        if sv.lower() == tv.lower():
                            continue
                        key = tuple(sorted([sv.lower(), tv.lower()]))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        examples.append((sv, tv))
                        matched = True
                        if len(examples) >= n_examples:
                            return examples
            if matched:
                break

        if matched:
            continue

        # Fallback: best label-like predicates
        if src_literals and tgt_literals:
            for _, src_vals in src_literals.items():
                for _, tgt_vals in tgt_literals.items():
                    for sv in src_vals:
                        for tv in tgt_vals:
                            if sv.lower() == tv.lower():
                                continue
                            key = tuple(sorted([sv.lower(), tv.lower()]))
                            if key in seen_pairs:
                                continue
                            seen_pairs.add(key)
                            examples.append((sv, tv))
                            if len(examples) >= n_examples:
                                return examples

    return examples


def print_transformation_result(val1, val2, result1, result2, idx, dataset_name):
    """Pretty print a transformation result."""
    print(f"\n{'─'*80}")
    print(f"[{dataset_name}] Example {idx}")
    print('─'*80)
    print(f"INPUT:")
    print(f"  Source: '{val1}'")
    print(f"  Target: '{val2}'")
    print(f"\nOUTPUT:")
    print(f"  Source: '{result1}'")
    print(f"  Target: '{result2}'")

    # Analyze transformation
    print(f"\nANALYSIS:")

    src_changed = result1 != val1
    tgt_changed = result2 != val2

    if src_changed:
        print(f"  ✓ Source changed")
    else:
        print(f"  ❌ Source unchanged (identical to input)")

    if tgt_changed:
        print(f"  ✓ Target changed")
    else:
        print(f"  ❌ Target unchanged (identical to input)")

    # Check for potential issues
    if len(result1) > len(val1) * 2.5:
        print(f"  ⚠️  Source output much longer than input (possible garbage)")
    if len(result2) > len(val2) * 2.5:
        print(f"  ⚠️  Target output much longer than input (possible garbage)")

    # Check similarity between outputs
    if result1.lower() == result2.lower():
        print(f"  ⚠️  Outputs are identical (not diverse)")
    elif result1.lower() in result2.lower() or result2.lower() in result1.lower():
        print(f"  ⚠️  One output contains the other")
    else:
        print(f"  ✓ Outputs are distinct")

    # Check if outputs are swapped
    if result1.lower() == val2.lower() and result2.lower() == val1.lower():
        print(f"  ⚠️  Outputs are simply swapped (no real transformation)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize augmentation by dataset")
    parser.add_argument("--datasets", nargs="+",
                        default=["BBC_DB", "D_W", "ICEWS_WIKI", "ICEWS_YAGO"],
                        help="Datasets to test")
    parser.add_argument("--examples-per-dataset", type=int, default=30,
                        help="Number of examples per dataset")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "augmentation" / "plm.yaml",
                        help="Config file")

    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    # Initialize BART with GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*80)
    print("DATASET AUGMENTATION VISUALIZATION")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")

    print("\n" + "-"*80)
    print("CURRENT PARAMETERS")
    print("-"*80)
    print(f"  model_name:           {bart_cfg.get('model_name')}")
    print(f"  base_alpha:           {bart_cfg.get('base_alpha')}")
    print(f"  alpha_spread:         {bart_cfg.get('alpha_spread')}")
    print(f"  temperature:          {gen_cfg.get('temperature')}")
    print(f"  top_p:                {gen_cfg.get('top_p')}")
    print(f"  num_beams:            {gen_cfg.get('num_beams')}")
    print(f"  repetition_penalty:   {gen_cfg.get('repetition_penalty')}")
    print(f"  noise_std:            {gen_cfg.get('noise_std')}")
    print(f"  enable_retry:         {bart_cfg.get('enable_retry_on_identical_tokens', True)}")
    print(f"  retry_threshold:      {gen_cfg.get('identical_tokens_threshold')}")

    print("\n\nInitializing BART model...")
    from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM

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

    print("✓ BART model loaded\n")

    # Process each dataset separately
    total_examples = 0
    dataset_stats = {}

    for dataset_name in args.datasets:
        print(f"\n\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print('='*80)

        examples = load_value_pairs_from_dataset(dataset_name, args.examples_per_dataset)

        if not examples:
            print(f"  ⚠️  No valid examples found for {dataset_name}")
            dataset_stats[dataset_name] = {"examples": 0, "transformed": 0}
            continue

        print(f"\n✓ Found {len(examples)} example pairs")

        transformed_count = 0

        # Test transformations
        for i, (val1, val2) in enumerate(examples, 1):
            total_examples += 1
            try:
                result1, result2 = interpolator.interpolate_pair(val1, val2)

                print_transformation_result(val1, val2, result1, result2, i, dataset_name)

                # Count as transformed if at least one output changed
                if result1 != val1 or result2 != val2:
                    transformed_count += 1

            except Exception as e:
                print(f"\n⚠️  Transformation {i} failed: {e}")

        dataset_stats[dataset_name] = {
            "examples": len(examples),
            "transformed": transformed_count
        }

    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for dataset_name, stats in dataset_stats.items():
        examples = stats["examples"]
        transformed = stats["transformed"]
        if examples > 0:
            pct = (transformed / examples) * 100
            print(f"\n{dataset_name}:")
            print(f"  Examples tested: {examples}")
            print(f"  Successfully transformed: {transformed} ({pct:.1f}%)")
        else:
            print(f"\n{dataset_name}: No examples found")

    print(f"\nTotal transformations tested: {total_examples}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
