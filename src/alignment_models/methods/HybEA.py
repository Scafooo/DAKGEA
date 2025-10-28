from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import logger

@MODEL_REGISTRY.register("hybea")
class HybEA:
    """
    Skeleton for the HybEA Entity Alignment model.
    This will run entity alignment between two KGs using hybrid embeddings.
    """

    def __init__(self, config):
        self.config = config
        logger.info("[HybEA] Model initialized.")

    def evaluate(self, dataset_reduced, dataset_augmented):
        """
        Dummy evaluation: returns placeholder scores.
        Replace with real HybEA invocation later.
        """
        logger.info(f"[HybEA] Evaluating on dataset with {len(dataset_reduced.aligned_entities)} entities.")
        # TODO: implement integration with HybEA repo
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
