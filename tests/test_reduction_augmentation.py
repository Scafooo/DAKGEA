from src.augmentation.methods.plm import PLMAugmenter
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
from src.augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.reduction.registry import load_builtin_reducers, REDUCTION_REGISTRY

import logging
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

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
# 4. PLM Augmentation
# ----------------------------------------------------------------------------
augmenter = PLMAugmenter({
    "augmentation": {
        "ratio": 0.5,
        "max_depth": 2,
        # Note: value_consistency config is loaded from config/augmentation/plm.yaml
        "bart": {
            "model_name": "facebook/bart-base",  # BART-base for 8GB GPU (optimal)
            "enable_finetuning": True,
            "force_retrain": False,  # Use existing model (noise injection works at runtime!)
            "out_dir": "./bart_plm_model_base",  # Use existing model
            "epochs": 10,  # Optimal epochs (early stopping at patience=3)
            "batch_size": 8,  # Maximum batch size for 8GB GPU
            "learning_rate": 5.0e-5,         # Learning rate
            "max_train_samples": 4000,  # Full training samples

            # Regularization (prevent overfitting)
            "weight_decay": 0.01,            # L2 regularization
            "warmup_steps": 100,             # LR warmup steps
            "max_grad_norm": 1.0,            # Gradient clipping
            "patience": 3,                   # Early stopping patience

            # BART interpolation parameters (OPTIMAL TUNED VALUES)
            "base_alpha": 0.5,        # Base interpolation weight (balanced)
            "alpha_spread": 0.50,     # Moderate spread for balanced mixing (reduced from 0.55)

            # Generation parameters (OPTIMAL TUNED VALUES - score: 0.924)
            "generation": {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_k": 0,              # Disabled (use top_p instead)
                "top_p": 0.9,            # Nucleus sampling (optimal from tuning)
                "temperature": 0.9,      # Sampling temperature (optimal from tuning)
                "num_beams": 2,          # Beam search (optimal from tuning)
                "repetition_penalty": 1.7,  # Repetition penalty (optimal from tuning)
                "length_penalty": 1.3,   # Neutral
                "no_repeat_ngram_size": 3,  # N-gram blocking (optimal from tuning)

                # Noise injection (moderate noise + moderate alpha_spread = balanced creativity)
                "enable_noise_injection": True,  # Enable noise injection
                "noise_std": 0.10,       # Base noise level
                "noise_apply_when": "identical_inputs",  # Only when source=target

                # Retry mechanism to avoid identical tokens
                "enable_retry_on_identical_tokens": True,  # Retry if output has identical tokens
                "max_retries": 100,        # Maximum retry attempts
                "noise_increment": 0.0,    # Increase noise by this amount on each retry (0.0 = keep fixed)
                "temperature_increment": 0.02,  # Increase temperature by this amount on each retry
                "identical_tokens_threshold": 0.3,  # Trigger retry only if >30% of tokens are identical
            },

            # Semantic predicate matching configuration
            "predicate_matching": {
                "similarity_threshold": 0.6,  # Lowered to find more matches
                "use_value_similarity": True,  # Enable hybrid matching (name + values)
                "name_weight": 0.85,  # Weight for name similarity (increased from 0.7)
                "value_weight": 0.15,  # Weight for value similarity (decreased from 0.3)
                "alignment_sample_size": 100,  # Entities to sample for alignment
            },

            # Unmatched attributes generation
            "generate_unmatched": True,  # Generate variations for non-matching attributes
            "unmatched_sample_rate": 1.0,  # Sample 100% of unmatched attributes (generate all)

            # Advanced training modules
            "advanced_training": {
                "stratified_sampling": {
                    "enable": False,
                    "min_samples_per_predicate": 5,
                    "max_samples_per_predicate": 5000,
                    "balancing_strategy": "sqrt"  # sqrt, uniform, or log
                },
                "advanced_noising": {
                    "enable": False,
                    "span_corruption_ratio": 0.2,  # 30% of spans corrupted
                    "mean_span_length": 3,  # Average span length to mask
                    "entity_aware_masking": True,  # Higher probability for entities
                    "entity_mask_prob": 0.7  # 50% probability to mask detected entities
                }
            }
        }
    },
    "experiment": {"seed": 11037}
})

dataset_augmented = augmenter.augment(dataset)



