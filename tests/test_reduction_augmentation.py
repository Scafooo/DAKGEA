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
            "model_name": "facebook/bart-base",  # Use smaller model for 8GB GPU
            "enable_finetuning": True,
            "force_retrain": True,  # Force retrain to apply new optimal parameters
            "out_dir": "./tests/bart_test_model",  # Percorso originale
            "epochs": 2,  # Ridotto per test veloci
            "batch_size": 4,  # Reduced batch size for 8GB GPU
            "max_train_samples": 1000,  # Limitato per test veloci

            # Regularization (prevent overfitting)
            "weight_decay": 0.01,            # L2 regularization
            "warmup_steps": 50,              # LR warmup (reduced for test)
            "max_grad_norm": 1.0,            # Gradient clipping

            # BART interpolation parameters (OPTIMAL TUNED VALUES)
            "base_alpha": 0.5,        # Base interpolation weight (balanced)
            "alpha_spread": 0.35,     # High variation for creative transformations

            # Generation parameters (OPTIMAL TUNED VALUES - score: 0.924)
            "generation": {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_k": 0,              # Disabled (use top_p instead)
                "top_p": 0.9,            # Nucleus sampling (optimal from tuning)
                "temperature": 1.0,      # Sampling temperature (optimal from tuning)
                "num_beams": 2,          # Beam search (optimal from tuning)
                "repetition_penalty": 1.7,  # Repetition penalty (optimal from tuning)
                "length_penalty": 1.0,   # Neutral
                "no_repeat_ngram_size": 3,  # N-gram blocking (optimal from tuning)
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



