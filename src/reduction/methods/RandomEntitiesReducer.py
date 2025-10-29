from src.logger import get_logger
from src.reduction.registry import REDUCTION_REGISTRY

logger = get_logger(__name__)

@REDUCTION_REGISTRY.register("random_entities")
class RandomEntitiesReducer:
    def __init__(self, config):
        self.config = config

    def reduce(self, dataset):
        target_entities = self.config["reduction"]["target_entities"]

        logger.info("Reducing dataset to ~%d aligned entities (random)", target_entities)

        return dataset
