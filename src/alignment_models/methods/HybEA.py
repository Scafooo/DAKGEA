from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import get_logger

logger = get_logger(__name__)

@MODEL_REGISTRY.register("hybea")
class HybEA:
    """
    Skeleton for the HybEA Entity Alignment model.
    This will run entity alignment between two KGs using hybrid embeddings.
    """

    def __init__(self, config):
        self.config = config
        logger.debug("[HybEA] Model initialized with config: %s", config)

    def evaluate(self, dataset_reduced, dataset_augmented):
        """
        Dummy evaluation: returns placeholder scores.
        Replace with real HybEA invocation later.
        """
        logger.info(
            "[HybEA] Evaluating on dataset with %d aligned entities.",
            len(dataset_reduced.aligned_entities),
        )
        # TODO: implement integration with HybEA repo
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
