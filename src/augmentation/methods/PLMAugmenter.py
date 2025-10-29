from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.dataset.Dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter:

    def __init__(self, config):
        self.config = config
        logger.debug("[PLMAugmenter] Initialized with config: %s", config)

    def augment(self, dataset: Dataset) -> Dataset:
        logger.info(
            "[PLMAugmenter] Augmenting dataset with %d aligned pairs.",
            len(dataset.aligned_entities),
        )
        # TODO: implement actual augmentation (e.g., latent interpolation)
        return dataset
