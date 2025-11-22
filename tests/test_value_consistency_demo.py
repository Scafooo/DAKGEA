"""Quick test to demonstrate value consistency in action with DEBUG logs."""

import logging
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure DEBUG logging to see value consistency messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress some noisy loggers
logging.getLogger("rdflib.term").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from src.augmentation.methods.plm import PLMAugmenter
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
from src.reduction.registry import load_builtin_reducers, REDUCTION_REGISTRY

print("="*80)
print("🔍 VALUE CONSISTENCY DEMONSTRATION")
print("="*80)
print("\nThis test will show value consistency in action.")
print("Look for log messages containing:")
print("  • [VALUE_CONSISTENCY]")
print("  • 'Reusing cached variation'")
print("  • 'Using inter-node value cache'")
print("\n" + "="*80)
print("LOADING DATASET...")
print("="*80 + "\n")

# Load dataset
reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/BBC_DB/attribute_data")

print("\n" + "="*80)
print("REDUCING DATASET (for quick test)...")
print("="*80 + "\n")

# Reduce to small size for quick test
load_builtin_reducers()
reducer = REDUCTION_REGISTRY.get("random_entities")(
    {"reduction": {"target_entities": 50}, "experiment": {"seed": 42}}
)
reducer.reduce(dataset)

print("\n" + "="*80)
print("STARTING PLM AUGMENTATION WITH VALUE CONSISTENCY...")
print("="*80)
print("\nConfiguration:")
print("  • Intra-node consistency: ENABLED")
print("  • Inter-node consistency: ENABLED")
print("  • Scope: alignment_pair")
print("  • Temperature: 1.5 (high creativity)")
print("  • Alpha spread: 0.25 (high variation)")
print("\n" + "="*80 + "\n")

# Configure augmenter with value consistency
augmenter = PLMAugmenter({
    "augmentation": {
        "ratio": 0.3,  # Small ratio for quick test
        "max_depth": 1,

        # VALUE CONSISTENCY ENABLED
        "value_consistency": {
            "intra_node": {
                "enabled": True,
                "selection": "first"
            },
            "inter_node": {
                "enabled": True,
                "scope": "alignment_pair"
            }
        },

        "bart": {
            "enable_finetuning": False,  # Skip training for quick test
            "base_alpha": 0.5,
            "alpha_spread": 0.25,
            "generation": {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_k": 0,
                "top_p": 0.90,
                "temperature": 1.5,
                "num_beams": 5,
                "repetition_penalty": 1.7,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 4,
            },
        }
    },
    "experiment": {"seed": 42}
})

# Run augmentation
dataset_augmented = augmenter.augment(dataset)

print("\n" + "="*80)
print("✅ AUGMENTATION COMPLETE!")
print("="*80)
print(f"\nResults:")
print(f"  • Original aligned pairs: {len(dataset.aligned_entities)}")
print(f"  • Source entities: {len(list(dataset.knowledge_graph_source.subjects(unique=True)))} → {len(list(dataset_augmented.knowledge_graph_source.subjects(unique=True)))}")
print(f"  • Target entities: {len(list(dataset.knowledge_graph_target.subjects(unique=True)))} → {len(list(dataset_augmented.knowledge_graph_target.subjects(unique=True)))}")
print("\n" + "="*80)
print("📝 SUMMARY")
print("="*80)
print("\nIf value consistency worked correctly, you should have seen:")
print("  ✓ [VALUE_CONSISTENCY] log messages about cache creation")
print("  ✓ 'Reusing cached variation' for duplicate values")
print("  ✓ Same values (like 'debbi peterson') getting same augmentation")
print("\n")
