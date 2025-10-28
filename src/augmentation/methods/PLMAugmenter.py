from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.dataset.Dataset import Dataset
from src.logger import logger


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter:

    def __init__(self, config):
        self.config = config
        logger.info("[PLMAugmenter] Initialized with config.")

    def augment(self, dataset: Dataset) -> Dataset:
        logger.info(f"[PLMAugmenter] Augmenting dataset with {len(dataset.aligned_entities)} aligned pairs.")
        # TODO: implement actual augmentation (e.g., latent interpolation)
        return dataset
