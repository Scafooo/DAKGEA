"""
PLM Augmentation Test - RTX 4090 Optimized
===========================================

This test leverages RTX 4090's 24GB VRAM for maximum quality results:
- BART-large (406M parameters)
- Large batch size (32)
- All training samples (no limit)
- Longer sequences (128 input / 64 output)
- More beams (4) for better generation quality

Expected training time: ~30-45 minutes
Expected results: Higher quality than BART-base with less gibberish
"""

from src.augmentation.methods.plm import PLMAugmenter
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
from src.augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.reduction.registry import load_builtin_reducers, REDUCTION_REGISTRY

import logging
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

print("=" * 80)
print("RTX 4090 OPTIMIZED - BART-LARGE AUGMENTATION")
print("=" * 80)
print("\nConfiguration:")
print("  • Model: facebook/bart-large (406M parameters)")
print("  • Batch size: 32 (vs 8 on RTX 4070)")
print("  • Epochs: 15 (with early stopping patience=5)")
print("  • Training samples: ALL (no limit)")
print("  • Sequence length: 128 input / 64 output")
print("  • Beams: 4 (vs 2 on RTX 4070)")
print("  • VRAM usage: ~12-14 GB (out of 24GB)")
print("\n" + "=" * 80 + "\n")

# ----------------------------------------------------------------------------
# 1. Dataset loading
# ----------------------------------------------------------------------------
reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/BBC_DB/attribute_data")

# ----------------------------------------------------------------------------
# 2. Reduction (optional)
# ----------------------------------------------------------------------------
load_builtin_reducers()
reducer = REDUCTION_REGISTRY.get("random_entities")(
    {"reduction": {"target_entities": 400}, "experiment": {"seed": 11037}}
)
reducer.reduce(dataset)

# ----------------------------------------------------------------------------
# 3. SetKnowledgeGraph creation
# ----------------------------------------------------------------------------
skg = SetKnowledgeGraph.from_dataset(dataset)

# ----------------------------------------------------------------------------
# 4. PLM Augmentation - RTX 4090 OPTIMIZED
# ----------------------------------------------------------------------------
augmenter = PLMAugmenter({
    "augmentation": {
        "ratio": 0.5,
        "max_depth": 2,

        # Value consistency (same as 8GB config)
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

        # RTX 4090 OPTIMIZED BART CONFIG
        "bart": {
            "model_name": "facebook/bart-large",  # 406M parameters (vs 140M base)
            "enable_finetuning": True,
            "force_retrain": False,  # Set to True for fresh training
            "out_dir": "./bart_plm_model_large",  # Separate dir for BART-large

            # Training parameters optimized for 24GB VRAM
            "epochs": 15,                    # More epochs (vs 10 on 8GB)
            "batch_size": 32,                # Large batch (vs 8 on 8GB) - 4x bigger!
            "learning_rate": 3.0e-5,         # Lower for BART-large
            "max_train_samples": None,       # Use ALL samples (no limit)

            # Regularization
            "weight_decay": 0.01,
            "warmup_steps": 500,             # More warmup for large batch
            "max_grad_norm": 1.0,
            "patience": 5,                   # Higher patience for BART-large

            # Interpolation parameters
            "base_alpha": 0.5,
            "alpha_spread": 0.45,            # Optimal from tuning
            "max_len_in": 128,               # Longer sequences (vs 96)
            "max_len_out": 64,               # Longer outputs (vs 48)

            # Generation parameters
            "generation": {
                "max_new_tokens": 48,        # More tokens (vs 32)
                "do_sample": True,
                "top_k": 0,
                "top_p": 0.9,                # Optimal from tuning
                "temperature": 1.0,          # Optimal from tuning
                "num_beams": 4,              # More beams (vs 2) for quality
                "repetition_penalty": 1.7,   # Optimal from tuning
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,   # Optimal from tuning

                # Noise injection - slightly lower for BART-large
                "enable_noise_injection": True,
                "noise_std": 0.18,           # Lower than 0.21 (BART-large is stronger)
                "noise_apply_when": "identical_inputs",
            },

            # Semantic predicate matching
            "predicate_matching": {
                "similarity_threshold": 0.6,
                "use_value_similarity": True,
                "name_weight": 0.85,
                "value_weight": 0.15,
                "alignment_sample_size": 200,  # More samples (vs 100)
            },

            # Unmatched attributes
            "generate_unmatched": True,
            "unmatched_sample_rate": 1.0,

            # Advanced training (can enable on 4090)
            "advanced_training": {
                "stratified_sampling": {
                    "enable": False,  # Can enable for better balance
                    "min_samples_per_predicate": 10,
                    "max_samples_per_predicate": 10000,
                    "balancing_strategy": "sqrt"
                },
                "advanced_noising": {
                    "enable": False,  # Can enable for better robustness
                    "span_corruption_ratio": 0.3,
                    "mean_span_length": 3,
                    "entity_aware_masking": True,
                    "entity_mask_prob": 0.5
                },
                "training_augmentation": {
                    "enable": True,   # Character-level noise augmentation
                    "random_noise": True,
                    "augmentation_ratio": 0.3
                }
            }
        }
    },
    "experiment": {"seed": 11037}
})

print("\n" + "=" * 80)
print("STARTING AUGMENTATION...")
print("=" * 80 + "\n")

dataset_augmented = augmenter.augment(dataset)

print("\n" + "=" * 80)
print("✓ AUGMENTATION COMPLETED")
print("=" * 80)
print("\nTo analyze results, run:")
print("  python tests/test_reduction_augmentation_4090.py 2>&1 | python tests/analyze_noise_results.py")
