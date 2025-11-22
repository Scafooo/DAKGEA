"""Test script to see value consistency in action with detailed logs."""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s',
    datefmt='%H:%M:%S'
)

from src.config.loader import load_yaml
from src.data.readers.factory import DatasetReaderFactory
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter


def main():
    """Run a small augmentation test to see value consistency logs."""
    print("\n" + "="*80)
    print("🔍 VALUE CONSISTENCY LOG TEST")
    print("="*80)

    # Load configuration
    config_path = project_root / "config" / "augmentation" / "plm.yaml"
    config = load_yaml(config_path)

    print(f"\n📋 Configuration:")
    print(f"   • Intra-node consistency: {config['augmentation']['value_consistency']['intra_node']['enabled']}")
    print(f"   • Inter-node consistency: {config['augmentation']['value_consistency']['inter_node']['enabled']}")
    print(f"   • Inter-node scope: {config['augmentation']['value_consistency']['inter_node']['scope']}")
    print(f"\n{'='*80}")
    print("📝 LOGS (watch for [VALUE_CONSISTENCY] and 'Reusing cached' messages)")
    print("="*80 + "\n")

    # Load small dataset
    dataset_name = "D_W_15K_V2"
    dataset_path = f"data/raw/openea/{dataset_name}"

    print(f"Loading dataset: {dataset_name}...")
    reader = DatasetReaderFactory.create_reader("openea")
    dataset = reader.read(dataset_path)

    # Override config for small test
    aug_config = config['augmentation'].copy()
    aug_config['ratio'] = 0.01  # Only 1% augmentation for quick test
    aug_config['max_depth'] = 1
    aug_config['bart']['enable_finetuning'] = False  # Skip training

    print(f"\n✓ Dataset loaded: {len(dataset.aligned_entities)} aligned pairs")
    print(f"✓ Running augmentation with ratio=0.01 (small test)...\n")

    # Run augmentation
    augmenter = PLMAugmenter(aug_config)
    augmented_dataset = augmenter.augment(dataset)

    print("\n" + "="*80)
    print(f"✅ AUGMENTATION COMPLETE")
    print("="*80)
    print(f"\n📊 Results:")
    print(f"   • Original entities (source): {len(dataset.knowledge_graph_source)}")
    print(f"   • Augmented entities (source): {len(augmented_dataset.knowledge_graph_source)}")
    print(f"   • Original entities (target): {len(dataset.knowledge_graph_target)}")
    print(f"   • Augmented entities (target): {len(augmented_dataset.knowledge_graph_target)}")

    print(f"\n💡 Look for these log patterns above:")
    print(f"   • [VALUE_CONSISTENCY] Using ... cache")
    print(f"   • [VALUE_CONSISTENCY] Created new ... cache")
    print(f"   • [VALUE_CONSISTENCY] Cleared ... cache")
    print(f"   • └─ Reusing cached variation for ...")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
