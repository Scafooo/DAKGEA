#!/usr/bin/env python3
"""Show transformation examples from actual datasets with current config."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rdflib import Literal
from src.logger import get_logger, set_global_level
from src.config.loader import load_yaml
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory

# Set minimal logging for clean output
set_global_level("WARNING")
logger = get_logger(__name__)

def load_examples_from_dataset(dataset_name: str, n_examples: int = 5):
    """Load real value pairs from a dataset.

    Args:
        dataset_name: Dataset name (e.g., "BBC_DB", "D_W", "ICEWS_WIKI", "ICEWS_YAGO")
        n_examples: Number of example pairs to extract

    Returns:
        List of (value1, value2) tuples from aligned entities
    """
    print(f"\nLoading examples from {dataset_name}...")

    reader = DatasetReaderFactory.create_reader("openea")
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "openea" / dataset_name

    if not raw_data_path.exists():
        print(f"  ⚠️  Dataset path not found: {raw_data_path}")
        return []

    dataset = reader.read(str(raw_data_path))
    aligned_entities = dataset.aligned_entities

    examples = []
    for src_uri, tgt_uri in list(aligned_entities)[:n_examples * 3]:  # Get more to ensure enough valid pairs
        src_kg = dataset.knowledge_graph_source
        tgt_kg = dataset.knowledge_graph_target

        # Get literal values from source entity
        src_values = []
        for _, _, obj in src_kg.triples((src_uri, None, None)):
            if isinstance(obj, Literal):
                val = str(obj)
                if 3 < len(val) < 100:
                    src_values.append(val)

        # Get literal values from target entity
        tgt_values = []
        for _, _, obj in tgt_kg.triples((tgt_uri, None, None)):
            if isinstance(obj, Literal):
                val = str(obj)
                if 3 < len(val) < 100:
                    tgt_values.append(val)

        # Create pairs from matching values
        for sv in src_values[:2]:
            for tv in tgt_values[:2]:
                if sv != tv and sv.strip() and tv.strip():
                    examples.append((sv, tv))
                    if len(examples) >= n_examples:
                        return examples

    return examples


def print_transformation(val1, val2, result1, result2, idx, total):
    """Pretty print a transformation result."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE {idx}/{total}")
    print('='*80)
    print(f"\n  INPUT:")
    print(f"    Source: {val1}")
    print(f"    Target: {val2}")
    print(f"\n  OUTPUT:")
    print(f"    Source: {result1}")
    print(f"    Target: {result2}")

    # Check if transformation worked
    print(f"\n  ANALYSIS:")

    src_changed = result1 != val1
    tgt_changed = result2 != val2

    if src_changed:
        print(f"    Source: ✓ Changed")
    else:
        print(f"    Source: ❌ No change (identical)")

    if tgt_changed:
        print(f"    Target: ✓ Changed")
    else:
        print(f"    Target: ❌ No change (identical)")

    # Check for garbage output
    if len(result1) > len(val1) * 3:
        print(f"    ⚠️  Source output much longer than input (possible garbage)")
    if len(result2) > len(val2) * 3:
        print(f"    ⚠️  Target output much longer than input (possible garbage)")

    # Check similarity between outputs
    if result1.lower() == result2.lower():
        print(f"    ⚠️  Outputs are identical")
    elif result1.lower() in result2.lower() or result2.lower() in result1.lower():
        print(f"    ⚠️  One output contains the other")
    else:
        print(f"    ✓ Outputs are distinct")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Show transformations from real datasets")
    parser.add_argument("--datasets", nargs="+", default=["BBC_DB", "D_W", "ICEWS_WIKI", "ICEWS_YAGO"],
                        help="Datasets to load examples from")
    parser.add_argument("--examples-per-dataset", type=int, default=2,
                        help="Number of examples to show per dataset")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "augmentation" / "plm.yaml",
                        help="Config file to use")

    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.get("generation", {})

    # Initialize BART with GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*80)
    print("DATASET TRANSFORMATION VISUALIZATION")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")

    print("\n" + "-"*80)
    print("CURRENT PARAMETERS")
    print("-"*80)
    print(f"  base_alpha:           {bart_cfg.get('base_alpha')}")
    print(f"  alpha_spread:         {bart_cfg.get('alpha_spread')}")
    print(f"  temperature:          {gen_cfg.get('temperature')}")
    print(f"  top_p:                {gen_cfg.get('top_p')}")
    print(f"  num_beams:            {gen_cfg.get('num_beams')}")
    print(f"  repetition_penalty:   {gen_cfg.get('repetition_penalty')}")
    print(f"  noise_std:            {gen_cfg.get('noise_std')}")
    print(f"  enable_retry:         {bart_cfg.get('enable_retry_on_identical_tokens')}")
    print(f"  retry_threshold:      {gen_cfg.get('identical_tokens_threshold')}")
    print(f"  sentence_level:       {bart_cfg.get('enable_sentence_level')}")

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

    # Load and test examples from each dataset
    total_examples = 0
    for dataset_name in args.datasets:
        examples = load_examples_from_dataset(dataset_name, args.examples_per_dataset)

        if not examples:
            print(f"  ⚠️  No examples found for {dataset_name}")
            continue

        print(f"  ✓ Loaded {len(examples)} examples from {dataset_name}")

        # Test transformations
        for i, (val1, val2) in enumerate(examples, 1):
            total_examples += 1
            try:
                # Temporarily enable INFO logging for this transformation
                set_global_level("INFO")
                result1, result2 = interpolator.interpolate_pair(val1, val2)
                set_global_level("WARNING")

                print_transformation(val1, val2, result1, result2, total_examples,
                                    len(args.datasets) * args.examples_per_dataset)
            except Exception as e:
                set_global_level("WARNING")
                print(f"\n⚠️  Transformation failed: {e}")

    print("\n" + "="*80)
    print(f"COMPLETED - Tested {total_examples} transformations")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
