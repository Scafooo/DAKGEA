import logging
import random
from typing import Set, Tuple
from rdflib import URIRef

from src.core.dataset import Dataset
from src.reduction.methods.random_entities.reducer_random_entities import RandomEntitiesReducer
from src.reduction.registry import REDUCTION_REGISTRY

logger = logging.getLogger(__name__)

@REDUCTION_REGISTRY.register("forget_labels")
class ForgetLabelsReducer(RandomEntitiesReducer):
    """
    Randomly reduces alignment labels but KEEPS the graph structure intact (prune_graphs=False).
    Used for 'Forget Labels' experiments where we test label efficiency.
    
    This implementation overrides RandomEntitiesReducer to skip the _prune_graphs step.
    It also correctly handles 'ratio' from configuration if 'target_entities' is not specified.
    """

    def __init__(self, config):
        super().__init__(config)
        reduction_cfg = self.config.get("reduction", {})
        self.ratio = reduction_cfg.get("ratio")
        
        # If target_entities is default (1) but ratio is provided, we will calculate target later
        # We store ratio to use it in reduce()

    def reduce(self, dataset: Dataset) -> Dataset:
        logger.info("[STEP] ForgetLabels (No-Prune) reduction started")
        aligned_set = self._normalise_alignment(dataset.aligned_entities)
        total_pairs = len(aligned_set)

        if total_pairs == 0:
            logger.warning("Dataset has no aligned entities; skipping reduction.")
            dataset.aligned_entities = aligned_set
            return dataset

        # Calculate target pairs based on ratio if available, otherwise use target_entities
        if self.ratio is not None:
            target_pairs = max(1, int(total_pairs * float(self.ratio)))
            logger.info(f"Calculated target pairs from ratio {self.ratio}: {target_pairs}")
        else:
            target_pairs = min(self.target_entities, total_pairs)

        remove_count = total_pairs - target_pairs

        logger.info(
            "Reducing dataset to %d aligned entity pairs (from %d) using random sampling. Graphs will remain UNCHANGED.",
            target_pairs,
            total_pairs,
        )
        logger.debug("Random reduction seed: %s", self.seed)

        source_size_before = len(dataset.knowledge_graph_source)
        target_size_before = len(dataset.knowledge_graph_target)

        if remove_count > 0:
            to_remove = self._sample_pairs_to_remove(aligned_set, remove_count, self.seed)
            logger.debug("Selected %d alignment pairs for removal (forgetting).", len(to_remove))
            
            # CRITICAL CHANGE: We do NOT call _prune_graphs here.
            # self._prune_graphs(dataset, to_remove) 
            
            remaining = aligned_set - to_remove
        else:
            logger.info(
                "Target pair count (%d) >= available pairs (%d); nothing to remove.",
                target_pairs,
                total_pairs,
            )
            remaining = aligned_set

        dataset.aligned_entities = remaining
        
        # We also skip _filter_alignment because we want to keep the graph intact.
        # if self.filter_alignment:
        #     dataset.aligned_entities = self._filter_alignment(dataset)

        source_size_after = len(dataset.knowledge_graph_source)
        target_size_after = len(dataset.knowledge_graph_target)

        logger.info(
            "Reduction complete. Source triples: %d → %d; target triples: %d → %d; aligned pairs: %d → %d.",
            source_size_before,
            source_size_after,
            target_size_before,
            target_size_after,
            total_pairs,
            len(dataset.aligned_entities),
        )
        logger.info("[SUCCESS] ForgetLabels reduction finished")

        return dataset