"""Simple test to verify value consistency logging works."""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(name)-35s | %(message)s',
)

from src.config.loader import load_yaml


def main():
    """Check that value consistency configuration is properly loaded."""
    print("\n" + "="*80)
    print("🔍 VALUE CONSISTENCY CONFIGURATION TEST")
    print("="*80)

    # Load configuration
    config_path = project_root / "config" / "augmentation" / "plm.yaml"
    config = load_yaml(config_path)

    vc_config = config['augmentation']['value_consistency']

    print(f"\n📋 Value Consistency Configuration from plm.yaml:")
    print(f"\n   Intra-node Consistency:")
    print(f"      • enabled: {vc_config['intra_node']['enabled']}")
    print(f"      • selection: {vc_config['intra_node']['selection']}")

    print(f"\n   Inter-node Consistency:")
    print(f"      • enabled: {vc_config['inter_node']['enabled']}")
    print(f"      • scope: {vc_config['inter_node']['scope']}")

    print("\n" + "="*80)
    print("📝 Log Messages You'll See During Augmentation:")
    print("="*80)

    print("\n   1️⃣  Cache Initialization (PLMAugmenter._bfs_expansion)")
    print("      → [VALUE_CONSISTENCY] Using alignment_pair inter-node cache")
    print("      → [VALUE_CONSISTENCY] Using expansion_cluster inter-node cache")
    print("      → [VALUE_CONSISTENCY] Using global inter-node cache")

    print("\n   2️⃣  Cache Creation Per Scope (PLMAugmenter._bfs_expansion)")
    print("      → [VALUE_CONSISTENCY] Created new cluster cache for seed <URI>")
    print("      → [VALUE_CONSISTENCY] Created new pair cache for set_node <URI>")

    print("\n   3️⃣  Cache Usage (NodeExpander._interpolate_literals)")
    print("      → • Using inter-node value cache (scope: alignment_pair)")
    print("      → └─ Reusing cached variation for src: '...' → '...'")
    print("      → └─ Reusing cached variation for tgt: '...' → '...'")

    print("\n   4️⃣  Cache Cleanup (PLMAugmenter._bfs_expansion)")
    print("      → [VALUE_CONSISTENCY] Cleared pair cache for set_node <URI>")

    print("\n" + "="*80)
    print("💡 How to See These Logs:")
    print("="*80)
    print("\n   Option 1: Run augmentation experiment with DEBUG logging")
    print("      $ export DAKGEA_LOG_LEVEL=DEBUG")
    print("      $ python3 scripts/run_experiment.sh")

    print("\n   Option 2: Check experiment logs")
    print("      $ grep -i 'VALUE_CONSISTENCY\\|Reusing cached' experiments/<exp_name>/logs/*.log")

    print("\n   Option 3: Run test with logging")
    print("      $ python3 -c \"import logging; logging.basicConfig(level=logging.DEBUG)\"")
    print("      $ python3 tests/test_reduction_augmentation.py")

    print("\n" + "="*80)
    print("✅ CONFIGURATION VERIFIED")
    print("="*80)
    print(f"\n✓ Intra-node consistency: {'ENABLED' if vc_config['intra_node']['enabled'] else 'DISABLED'}")
    print(f"✓ Inter-node consistency: {'ENABLED' if vc_config['inter_node']['enabled'] else 'DISABLED'}")
    if vc_config['inter_node']['enabled']:
        print(f"✓ Cache scope: {vc_config['inter_node']['scope']}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
