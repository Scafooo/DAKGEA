from src.reduction.registry import REDUCTION_REGISTRY

@REDUCTION_REGISTRY.register("random_entities")
class RandomEntitiesReducer:
    def __init__(self, config):
        self.config = config

    def reduce(self, dataset):
        target_entities = self.config["reduction"]["target_entities"]

        from src.logger import logger
        logger.info(f"Reducing dataset to ~{target_entities} aligned entities (random)")

        return dataset
