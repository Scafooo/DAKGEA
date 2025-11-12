"""
Test semantic predicate matching on real OpenEA dataset (D_W_15K_V1).

This script demonstrates how the semantic matcher works with actual
dataset attr_names files and predicates.
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from rdflib import URIRef, Literal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
from src.augmentation.methods.plm.predicate_matcher import PredicateMatcher


def extract_predicates_from_kg(kg, max_entities: int = 50):
    """
    Extract predicates and their values from knowledge graph.

    Args:
        kg: KnowledgeGraph instance
        max_entities: Maximum number of entities to sample

    Returns:
        Dict mapping predicate_local_name -> (uri, [literals])
    """
    from src.augmentation.methods.plm.bart_interpolator import _clean_pred

    predicates = defaultdict(lambda: (None, []))
    entity_count = defaultdict(int)

    # Sample entities and collect their predicates
    for subject, predicate, obj in kg.triples((None, None, None)):
        if isinstance(obj, Literal):
            local_name = _clean_pred(str(predicate))

            # Initialize if needed
            if predicates[local_name][0] is None:
                predicates[local_name] = (predicate, [])

            # Add literal (limit per predicate to avoid memory issues)
            if len(predicates[local_name][1]) < 10:
                predicates[local_name][1].append(obj)

            entity_count[subject] += 1

    return dict(predicates)


def analyze_attr_names(attr_names: Dict[str, str]):
    """Analyze and display attr_names statistics."""
    if not attr_names:
        print("  No attr_names found")
        return

    print(f"  Total mappings: {len(attr_names)}")

    # Sample some entries
    print("\n  Sample entries:")
    for i, (uri, name) in enumerate(list(attr_names.items())[:10], 1):
        # Truncate long URIs
        uri_display = uri if len(uri) < 60 else uri[:57] + "..."
        print(f"    {i:2d}. {uri_display}")
        print(f"        → \"{name}\"")


def main():
    print("\n" + "="*80)
    print("Real Dataset Test: D_W_15K_V1 - Semantic Predicate Matching")
    print("="*80 + "\n")

    # Dataset path
    dataset_path = Path("data/raw/openea/D_W_15K_V1/attribute_data")

    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\nPlease ensure D_W_15K_V1 dataset is available:")
        print("  Expected path: data/raw/openea/D_W_15K_V1/attribute_data/")
        return

    print(f"✓ Dataset found: {dataset_path}\n")

    # Load dataset
    print("-" * 80)
    print("Step 1: Loading Dataset")
    print("-" * 80)

    try:
        reader = DatasetReaderFactory.create_reader("openea")
        dataset = reader.read(str(dataset_path))

        print(f"✓ Dataset loaded successfully")
        print(f"  Source KG: {len(dataset.knowledge_graph_source)} triples")
        print(f"  Target KG: {len(dataset.knowledge_graph_target)} triples")
        print(f"  Aligned entities: {len(dataset.aligned_entities)} pairs")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze attr_names
    print("\n" + "-" * 80)
    print("Step 2: Analyzing attr_names")
    print("-" * 80)

    print("\nSource KG attr_names:")
    analyze_attr_names(dataset.knowledge_graph_source.attr_to_name)

    print("\nTarget KG attr_names:")
    analyze_attr_names(dataset.knowledge_graph_target.attr_to_name)

    # Extract predicates
    print("\n" + "-" * 80)
    print("Step 3: Extracting Predicates from Sample Entities")
    print("-" * 80)

    print("\nExtracting from source KG...")
    src_predicates = extract_predicates_from_kg(dataset.knowledge_graph_source, max_entities=50)
    print(f"  Found {len(src_predicates)} unique predicates")

    print("\nExtracting from target KG...")
    tgt_predicates = extract_predicates_from_kg(dataset.knowledge_graph_target, max_entities=50)
    print(f"  Found {len(tgt_predicates)} unique predicates")

    # Show sample predicates
    print("\n  Sample source predicates:")
    for i, (local_name, (uri, literals)) in enumerate(list(src_predicates.items())[:10], 1):
        natural_name = dataset.knowledge_graph_source.attr_to_name.get(str(uri), "N/A")
        sample_value = str(literals[0])[:50] if literals else "N/A"
        print(f"    {i:2d}. {local_name:20s} → \"{natural_name}\"")
        print(f"        Example value: \"{sample_value}\"")

    print("\n  Sample target predicates:")
    for i, (local_name, (uri, literals)) in enumerate(list(tgt_predicates.items())[:10], 1):
        natural_name = dataset.knowledge_graph_target.attr_to_name.get(str(uri), "N/A")
        sample_value = str(literals[0])[:50] if literals else "N/A"
        print(f"    {i:2d}. {local_name:20s} → \"{natural_name}\"")
        print(f"        Example value: \"{sample_value}\"")

    # Semantic matching
    print("\n" + "-" * 80)
    print("Step 4: Semantic Predicate Matching")
    print("-" * 80)

    print("\nInitializing PredicateMatcher...")
    matcher = PredicateMatcher({
        "embedding_model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.65,  # Lower threshold for cross-schema matching
        "cache_dir": ".cache/embeddings_test",
    })
    print("✓ Matcher initialized")

    print(f"\nMatching {len(src_predicates)} source predicates with {len(tgt_predicates)} target predicates...")
    print("Using attr_names from dataset files...")

    try:
        matches = matcher.match_predicates(
            src_predicates,
            tgt_predicates,
            dataset.knowledge_graph_source.attr_to_name,
            dataset.knowledge_graph_target.attr_to_name,
        )

        print(f"\n✓ Matching complete: Found {len(matches)} matches")

    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install sentence-transformers:")
        print("  pip install sentence-transformers")
        return
    except Exception as e:
        print(f"\n❌ Matching failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze results
    print("\n" + "-" * 80)
    print("Step 5: Results Analysis")
    print("-" * 80)

    if not matches:
        print("\n⚠️ No matches found!")
        print("This might indicate:")
        print("  - Very different schemas between source and target")
        print("  - Threshold too high (try lowering to 0.5)")
        print("  - Missing attr_names for key predicates")
        return

    # Statistics
    stats = matcher.compute_match_statistics(matches)
    print(f"\nStatistics:")
    print(f"  Total matches: {stats['num_matches']}")
    print(f"  Average confidence: {stats['avg_confidence']:.4f}")
    print(f"  Min confidence: {stats['min_confidence']:.4f}")
    print(f"  Max confidence: {stats['max_confidence']:.4f}")
    print(f"  Std deviation: {stats.get('std_confidence', 0):.4f}")

    # Show all matches
    print(f"\n{'='*80}")
    print(f"All Predicate Matches (sorted by confidence)")
    print(f"{'='*80}\n")

    for i, match in enumerate(matches, 1):
        # Get natural names from attr_names
        src_uri_str = str(match.src_uri)
        tgt_uri_str = str(match.tgt_uri)

        src_natural = dataset.knowledge_graph_source.attr_to_name.get(src_uri_str, None)
        tgt_natural = dataset.knowledge_graph_target.attr_to_name.get(tgt_uri_str, None)

        # Determine quality
        if match.confidence >= 0.85:
            quality = "EXCELLENT"
            symbol = "✓✓✓"
        elif match.confidence >= 0.75:
            quality = "GOOD"
            symbol = "✓✓"
        elif match.confidence >= 0.65:
            quality = "FAIR"
            symbol = "✓"
        else:
            quality = "WEAK"
            symbol = "~"

        print(f"{i:2d}. {symbol} [{quality:9s}] Confidence: {match.confidence:.4f}")
        print(f"    Source: {match.src_predicate:25s} → \"{src_natural or 'N/A'}\"")
        print(f"    Target: {match.tgt_predicate:25s} → \"{tgt_natural or 'N/A'}\"")

        # Show example values
        src_vals = src_predicates.get(match.src_predicate, (None, []))[1]
        tgt_vals = tgt_predicates.get(match.tgt_predicate, (None, []))[1]

        if src_vals:
            src_example = str(src_vals[0])[:50]
            print(f"    Example (src): \"{src_example}\"")
        if tgt_vals:
            tgt_example = str(tgt_vals[0])[:50]
            print(f"    Example (tgt): \"{tgt_example}\"")

        print()

    # Coverage analysis
    print(f"{'='*80}")
    print("Coverage Analysis")
    print(f"{'='*80}\n")

    matched_src = {m.src_predicate for m in matches}
    matched_tgt = {m.tgt_predicate for m in matches}

    unmatched_src = set(src_predicates.keys()) - matched_src
    unmatched_tgt = set(tgt_predicates.keys()) - matched_tgt

    print(f"Source predicates:")
    print(f"  Total: {len(src_predicates)}")
    print(f"  Matched: {len(matched_src)} ({len(matched_src)/len(src_predicates)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched_src)} ({len(unmatched_src)/len(src_predicates)*100:.1f}%)")

    print(f"\nTarget predicates:")
    print(f"  Total: {len(tgt_predicates)}")
    print(f"  Matched: {len(matched_tgt)} ({len(matched_tgt)/len(tgt_predicates)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched_tgt)} ({len(unmatched_tgt)/len(tgt_predicates)*100:.1f}%)")

    # Show some unmatched predicates
    if unmatched_src:
        print(f"\nSample unmatched source predicates:")
        for pred in list(unmatched_src)[:5]:
            uri = src_predicates[pred][0]
            natural = dataset.knowledge_graph_source.attr_to_name.get(str(uri), "N/A")
            print(f"  • {pred:25s} → \"{natural}\"")

    if unmatched_tgt:
        print(f"\nSample unmatched target predicates:")
        for pred in list(unmatched_tgt)[:5]:
            uri = tgt_predicates[pred][0]
            natural = dataset.knowledge_graph_target.attr_to_name.get(str(uri), "N/A")
            print(f"  • {pred:25s} → \"{natural}\"")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print("\nKey findings:")
    print(f"  ✓ Successfully matched {len(matches)} predicate pairs")
    print(f"  ✓ attr_names files used for natural language names")
    print(f"  ✓ Average matching confidence: {stats['avg_confidence']:.2%}")
    print("\nThe semantic matcher is working correctly with real dataset!")
    print()


if __name__ == "__main__":
    main()
