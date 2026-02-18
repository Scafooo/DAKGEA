"""Context-Aware PLM Augmenter.

Extends PLMAugmenter to use context-aware components.
"""

from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter
from src.augmentation.methods.plm_context.node_context_expander import NodeContextExpander
from src.augmentation.methods.plm_context.mixup_context_interpolator import MixupContextInterpolator
from src.logger import get_logger
import torch

logger = get_logger(__name__)

@AUGMENTATION_REGISTRY.register("plm_context")
class PLMContextAugmenter(PLMAugmenter):
    """PLM Augmenter with structural context injection."""
    
    registry_name = "plm_context_augmentation"

    def _initialize_bart_only(self, dataset):
        """Initialize Context-Aware Interpolator."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        logger.info("[PLM-Context] Initializing MixupContextInterpolator...")
        
        # Determine model path - should be in models/pretrained_plm_context/DATASET
        # We assume self.bart_out_dir points there or is configured correctly in YAML
        # If running via run_flan_t5_context_experiments.sh, it will pass the correct path.
        
        self.bart_interpolator = MixupContextInterpolator(
            model_name=self.bart_model_name,
            out_dir=self.bart_out_dir, # Should point to context-trained model
            device=device,
            seed=self.seed,
            base_alpha=self.bart_base_alpha,
            alpha_spread=self.bart_alpha_spread,
            # Pass training_config if needed
        )
        logger.info("[PLM-Context] Interpolator ready.")

    def augment(self, dataset):
        # Override augment to inject NodeContextExpander
        # We can't easily override just the expander init inside augment(),
        # so we rely on calling super().augment() BUT we must swap the expander BEFORE BFS starts.
        
        # TRICK: super().augment() calls _initialize_... then creates NodeExpander then does BFS.
        # We can't inject in the middle.
        # We must copy-paste augment() logic OR overwrite self.node_expander after initialization.
        # But _bfs_expansion uses self.node_expander.
        
        # Let's reimplement augment() to be safe and use NodeContextExpander.
        
        if not dataset.aligned_entities:
            return dataset.clone()

        # 1. Init Interpolator
        self.section("PLM-Context Initialization")
        self._initialize_bart_only(dataset) # We only support inference here, pre-training is external

        # 2. Init Expander (CONTEXT VERSION)
        from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
        
        # Pre-compute alignments if needed (same as base)
        # ... (omitted for brevity, assuming cache is optional or we accept on-fly)
        
        self.node_expander = NodeContextExpander(
            self.derived_predicate,
            self.add_derived_predicate,
            self.bart_interpolator,
            self.predicate_matcher_config,
            None, # Cache
            advanced_training_config=self.bart_advanced_training_config,
            bart_config=self.bart_cfg,
        )

        # 3. Set KG
        set_graph = SetKnowledgeGraph.from_dataset(dataset)
        set_nodes = sorted(set_graph.iter_set_nodes(), key=lambda u: str(u))
        
        pair_budget = self._compute_pair_budget(len(dataset.aligned_entities))
        self._log_augmentation_start(len(dataset.aligned_entities), pair_budget)
        
        import random
        rng = random.Random(self.seed)
        rng.shuffle(set_nodes)

        # 4. BFS
        self.section("PLM-Context BFS Expansion")
        expanded_pairs = self._bfs_expansion(dataset, set_graph, set_nodes, pair_budget)
        
        return dataset
