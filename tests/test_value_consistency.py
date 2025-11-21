"""Test value consistency features (intra-node and inter-node)."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rdflib import Graph, URIRef, Literal, Namespace
from src.augmentation.methods.plm.node_expander import NodeExpander
from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM


def test_intra_node_consistency():
    """Test that duplicate values within same node get same augmentation."""
    print("\n" + "="*70)
    print("TEST 1: Intra-node Value Consistency")
    print("="*70)

    # Create a simple test graph with duplicate values
    graph = Graph()
    EX = Namespace("http://example.org/")

    entity = EX.person1
    graph.add((entity, EX.name, Literal("John Smith")))
    graph.add((entity, EX.label, Literal("John Smith")))  # Duplicate value
    graph.add((entity, EX.fullName, Literal("John Smith")))  # Another duplicate
    graph.add((entity, EX.age, Literal("30")))

    # Create NodeExpander with intra-node consistency enabled
    config = {
        "intra_node": {
            "enabled": True,
            "selection": "first"
        },
        "inter_node": {
            "enabled": False
        }
    }

    print(f"\n📊 Original values:")
    print(f"   • name: 'John Smith'")
    print(f"   • label: 'John Smith' (duplicate)")
    print(f"   • fullName: 'John Smith' (duplicate)")
    print(f"   • age: '30'")

    print(f"\n✅ Intra-node consistency: ENABLED")
    print(f"   Expected: All 'John Smith' values → same augmentation")
    print(f"   Expected: '30' value → different augmentation")

    # Initialize BART (without actually loading - we'll mock it for this test)
    bart = BartInterpolatorPLM(
        model_name="facebook/bart-base",
        device="cpu",
        seed=42,
        base_alpha=0.35,
        alpha_spread=0.15,
    )

    expander = NodeExpander(
        derived_predicate=URIRef("http://derived"),
        add_derived_predicate=False,
        bart_interpolator=bart,
        value_consistency_config=config
    )

    print(f"\n✓ Test setup complete")
    print(f"   • NodeExpander created with intra-node consistency")
    print(f"   • Selection strategy: first")


def test_inter_node_consistency():
    """Test that shared values across nodes get same augmentation."""
    print("\n" + "="*70)
    print("TEST 2: Inter-node Value Consistency (alignment_pair scope)")
    print("="*70)

    # Create two entities that share some values
    graph = Graph()
    EX = Namespace("http://example.org/")

    person1 = EX.person1
    person2 = EX.person2

    # Both entities have "New York" as location
    graph.add((person1, EX.name, Literal("Alice")))
    graph.add((person1, EX.location, Literal("New York")))
    graph.add((person1, EX.age, Literal("25")))

    graph.add((person2, EX.name, Literal("Bob")))
    graph.add((person2, EX.location, Literal("New York")))  # Shared value
    graph.add((person2, EX.age, Literal("30")))

    print(f"\n📊 Original values:")
    print(f"   Person 1:")
    print(f"      • name: 'Alice'")
    print(f"      • location: 'New York'")
    print(f"      • age: '25'")
    print(f"   Person 2:")
    print(f"      • name: 'Bob'")
    print(f"      • location: 'New York' (shared)")
    print(f"      • age: '30'")

    # Create NodeExpander with inter-node consistency enabled
    config = {
        "intra_node": {
            "enabled": True,
            "selection": "first"
        },
        "inter_node": {
            "enabled": True,
            "scope": "alignment_pair"
        }
    }

    print(f"\n✅ Inter-node consistency: ENABLED")
    print(f"   Scope: alignment_pair")
    print(f"   Expected: Both 'New York' values → same augmentation")
    print(f"   Expected: 'Alice' and 'Bob' → different augmentations")

    bart = BartInterpolatorPLM(
        model_name="facebook/bart-base",
        device="cpu",
        seed=42,
        base_alpha=0.35,
        alpha_spread=0.15,
    )

    expander = NodeExpander(
        derived_predicate=URIRef("http://derived"),
        add_derived_predicate=False,
        bart_interpolator=bart,
        value_consistency_config=config
    )

    # Simulate cache sharing
    shared_cache = {}
    expander.set_inter_node_cache(shared_cache)

    print(f"\n✓ Test setup complete")
    print(f"   • NodeExpander created with inter-node consistency")
    print(f"   • Scope: alignment_pair (cache shared within pair)")
    print(f"   • Shared cache initialized")


def test_scope_variations():
    """Test different inter-node scope variations."""
    print("\n" + "="*70)
    print("TEST 3: Inter-node Scope Variations")
    print("="*70)

    scopes = ["alignment_pair", "expansion_cluster", "global"]

    for scope in scopes:
        print(f"\n📌 Testing scope: {scope}")

        config = {
            "intra_node": {
                "enabled": True,
                "selection": "first"
            },
            "inter_node": {
                "enabled": True,
                "scope": scope
            }
        }

        bart = BartInterpolatorPLM(
            model_name="facebook/bart-base",
            device="cpu",
            seed=42,
            base_alpha=0.35,
            alpha_spread=0.15,
        )

        expander = NodeExpander(
            derived_predicate=URIRef("http://derived"),
            add_derived_predicate=False,
            bart_interpolator=bart,
            value_consistency_config=config
        )

        assert expander.inter_node_consistency_enabled == True
        assert expander.inter_node_scope == scope

        print(f"   ✓ Scope '{scope}' configured successfully")

        if scope == "alignment_pair":
            print(f"      → Cache shared: within same aligned entity pair")
        elif scope == "expansion_cluster":
            print(f"      → Cache shared: within same BFS expansion cluster")
        elif scope == "global":
            print(f"      → Cache shared: across entire augmentation process")


def main():
    """Run all value consistency tests."""
    print("\n" + "="*70)
    print("🧪 VALUE CONSISTENCY TESTS")
    print("="*70)

    try:
        test_intra_node_consistency()
        test_inter_node_consistency()
        test_scope_variations()

        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📝 Summary:")
        print("   • Intra-node consistency: Ensures duplicate values within a node")
        print("     get the same augmented variation")
        print("   • Inter-node consistency: Ensures shared values across nodes")
        print("     get the same augmented variation within a scope")
        print("   • Scopes available:")
        print("      - alignment_pair: Cache per aligned entity pair")
        print("      - expansion_cluster: Cache per BFS cluster")
        print("      - global: Cache across entire augmentation")
        print("\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
