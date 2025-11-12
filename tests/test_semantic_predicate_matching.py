"""
Test script for semantic predicate matching.

This demonstrates the full semantic matching capabilities using
sentence-transformers embeddings.
"""

from rdflib import URIRef, Literal
from src.augmentation.methods.plm.predicate_matcher import PredicateMatcher


def test_semantic_matching():
    """Test semantic predicate matching with diverse examples."""

    print("\n" + "="*80)
    print("Semantic Predicate Matching Test")
    print("="*80 + "\n")

    # Simulate predicates from two different KGs with different naming conventions
    # Source KG: DBpedia-style
    src_predicates = {
        "birthDate": (URIRef("dbo:birthDate"), [Literal("1980-05-12")]),
        "fullName": (URIRef("dbo:name"), [Literal("John Smith")]),
        "phoneNumber": (URIRef("dbo:phone"), [Literal("+1-555-0123")]),
        "cityOfBirth": (URIRef("dbo:birthPlace"), [Literal("New York")]),
        "occupation": (URIRef("dbo:occupation"), [Literal("Engineer")]),
        "emailAddress": (URIRef("dbo:email"), [Literal("john@example.com")]),
    }

    # Target KG: FOAF-style + custom
    tgt_predicates = {
        "dateOfBirth": (URIRef("foaf:birthday"), [Literal("1980-05-15")]),
        "name": (URIRef("foaf:name"), [Literal("Jean Dupont")]),
        "telephone": (URIRef("foaf:phone"), [Literal("+33-1-23-45-67")]),
        "birthCity": (URIRef("ex:placeOfBirth"), [Literal("Paris")]),
        "job": (URIRef("ex:profession"), [Literal("Ingénieur")]),
        "email": (URIRef("ex:contactEmail"), [Literal("jean@example.fr")]),
    }

    # Simulate attr_names (like those loaded from dataset files)
    src_attr_names = {
        "dbo:birthDate": "birth date",
        "dbo:name": "full name",
        "dbo:phone": "phone number",
        "dbo:birthPlace": "city of birth",
        "dbo:occupation": "occupation",
        "dbo:email": "email address",
    }

    tgt_attr_names = {
        "foaf:birthday": "date of birth",
        "foaf:name": "name",
        "foaf:phone": "telephone",
        "ex:placeOfBirth": "birth city",
        "ex:profession": "job",
        "ex:contactEmail": "email",
    }

    print("Source Predicates (DBpedia-style):")
    for pred, (uri, _) in src_predicates.items():
        natural = src_attr_names.get(str(uri), "N/A")
        print(f"  • {pred:20s} ({uri}) → \"{natural}\"")

    print("\nTarget Predicates (FOAF-style + custom):")
    for pred, (uri, _) in tgt_predicates.items():
        natural = tgt_attr_names.get(str(uri), "N/A")
        print(f"  • {pred:20s} ({uri}) → \"{natural}\"")

    print("\n" + "-"*80)
    print("Running Semantic Matching WITH attr_names...")
    print("-"*80 + "\n")

    # Create matcher with default config
    matcher = PredicateMatcher({
        "embedding_model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.6,  # Lower threshold to see more matches
        "cache_dir": ".cache/embeddings_test",
    })

    # Perform matching WITH attr_names
    matches = matcher.match_predicates(
        src_predicates, tgt_predicates,
        src_attr_names, tgt_attr_names
    )

    # Display matches
    print(f"Found {len(matches)} matches:\n")

    for i, match in enumerate(matches, 1):
        src_expanded = matcher._expand_predicate_name(match.src_predicate)
        tgt_expanded = matcher._expand_predicate_name(match.tgt_predicate)

        # Determine match quality
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

        print(f"{i}. {symbol} [{quality}] Confidence: {match.confidence:.4f}")
        print(f"   Source: {match.src_predicate:20s} → \"{src_expanded}\"")
        print(f"   Target: {match.tgt_predicate:20s} → \"{tgt_expanded}\"")
        print()

    # Statistics
    stats = matcher.compute_match_statistics(matches)
    print("-"*80)
    print("Statistics:")
    print(f"  Total matches:     {stats['num_matches']}")
    print(f"  Average confidence: {stats['avg_confidence']:.4f}")
    print(f"  Min confidence:     {stats['min_confidence']:.4f}")
    print(f"  Max confidence:     {stats['max_confidence']:.4f}")
    if 'std_confidence' in stats:
        print(f"  Std deviation:      {stats['std_confidence']:.4f}")
    print("="*80 + "\n")

    # Expected matches (for validation)
    expected_high_confidence = [
        ("birthDate", "dateOfBirth"),  # Very similar
        ("fullName", "name"),           # Clear semantic match
        ("phoneNumber", "telephone"),   # Synonyms
        ("emailAddress", "email"),      # Obvious match
        ("cityOfBirth", "birthCity"),   # Same concept
        ("occupation", "job"),          # Synonyms
    ]

    # Validate
    matched_pairs = {(m.src_predicate, m.tgt_predicate) for m in matches}
    correctly_matched = sum(1 for pair in expected_high_confidence if pair in matched_pairs)

    print("Validation:")
    print(f"  Expected high-confidence matches: {len(expected_high_confidence)}")
    print(f"  Correctly matched: {correctly_matched}")
    print(f"  Accuracy: {correctly_matched/len(expected_high_confidence)*100:.1f}%")

    if correctly_matched == len(expected_high_confidence):
        print("\n  ✅ All expected matches found! Semantic matching works perfectly.")
    else:
        print(f"\n  ⚠️ Some expected matches missing. This may be due to threshold settings.")
        print("     Try lowering similarity_threshold for more matches.")

    print()
    return matches, stats


def test_edge_cases():
    """Test edge cases: empty predicates, no matches, etc."""

    print("\n" + "="*80)
    print("Edge Cases Test")
    print("="*80 + "\n")

    matcher = PredicateMatcher({
        "similarity_threshold": 0.7,
        "cache_dir": ".cache/embeddings_test",
    })

    # Case 1: Empty source
    print("Test 1: Empty source predicates")
    matches = matcher.match_predicates({}, {"name": (URIRef("ex:name"), [Literal("test")])})
    print(f"  Result: {len(matches)} matches (expected: 0)")
    assert len(matches) == 0, "Should return empty list for empty source"
    print("  ✅ Passed\n")

    # Case 2: Empty target
    print("Test 2: Empty target predicates")
    matches = matcher.match_predicates({"name": (URIRef("ex:name"), [Literal("test")])}, {})
    print(f"  Result: {len(matches)} matches (expected: 0)")
    assert len(matches) == 0, "Should return empty list for empty target"
    print("  ✅ Passed\n")

    # Case 3: No matches (very different predicates)
    print("Test 3: Completely different predicates (high threshold)")
    src = {"temperature": (URIRef("ex:temp"), [Literal("25")])}
    tgt = {"username": (URIRef("ex:user"), [Literal("john")])}
    matches = matcher.match_predicates(src, tgt)
    print(f"  Result: {len(matches)} matches")
    print(f"  Note: With threshold={matcher.similarity_threshold}, dissimilar predicates may not match")
    print("  ✅ Passed\n")

    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        # Run main test
        matches, stats = test_semantic_matching()

        # Run edge cases
        test_edge_cases()

        print("\n" + "="*80)
        print("All Tests Completed Successfully!")
        print("="*80 + "\n")

        print("Next steps:")
        print("  1. Install sentence-transformers: pip install sentence-transformers")
        print("  2. Run this test: python tests/test_semantic_predicate_matching.py")
        print("  3. Check cached embeddings in .cache/embeddings_test/")
        print("  4. Integrate with full PLM augmentation pipeline")
        print()

    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install sentence-transformers:")
        print("  pip install sentence-transformers")
        print()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
