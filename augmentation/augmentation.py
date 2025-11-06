import os

import torch

from src.alignment_models.methods.hybea import legacy_config as cfg
from augmentation.PLMAugmenter import PLMAugmenter
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory
from src.logger import get_logger

logger = get_logger(__name__)


def run_augmentation():
    """Legacy entry point maintained for backward compatibility with older scripts."""
    read_dir_path = os.path.join(cfg.REDUCED_DATA_DIR, "attribute_data", cfg.DATASET)
    dataset_reader = DatasetReaderFactory.create_reader(cfg.MODEL)
    dataset = dataset_reader.read(read_dir_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {
        "experiment": {"seed": 42},
        "augmentation": {
            "plm_augmentation": {
                "device": str(device),
                "seed": 42,
                "output_dir": "./bart_attribute_plm",
                "aug_percentage": 0.5,
                "max_depth": 2,
            }
        },
    }

    augmenter = PLMAugmenter(config)
    dataset_augmented = augmenter.augment(dataset)

    write_att_dir_path = os.path.join(cfg.AUGMENTED_DATA_DIR, "attribute_data", cfg.DATASET)
    os.makedirs(write_att_dir_path, exist_ok=True)
    dataset_att_writer = DatasetWriterFactory.create_writer(cfg.MODEL)
    dataset_att_writer.write(dataset_augmented, write_att_dir_path)

    write_kn_dir_path = os.path.join(cfg.AUGMENTED_DATA_DIR, "knowformer_data", cfg.DATASET)
    os.makedirs(write_kn_dir_path, exist_ok=True)
    dataset_kn_writer = DatasetWriterFactory.create_writer(cfg.MODEL)
    dataset_kn_writer.write(dataset_augmented, write_kn_dir_path)


if __name__ == "__main__":
    run_augmentation()
