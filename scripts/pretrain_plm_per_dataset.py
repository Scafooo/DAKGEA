#!/usr/bin/env python3
"""Pre-train PLM (FLAN-T5-XL) model for each dataset.

This script fine-tunes a FLAN-T5-XL model with LoRA for each dataset
and saves the pre-trained model to be reused across all experiments.

Usage:
    python scripts/pretrain_plm_per_dataset.py --datasets BBC_DB D_W_15K_V1
    python scripts/pretrain_plm_per_dataset.py --all
    python scripts/pretrain_plm_per_dataset.py --datasets BBC_DB --dry-run

Output:
    models/pretrained_plm/{dataset}/
        ├── adapter_model.safetensors
        ├── adapter_config.json
        ├── tokenizer.json
        └── ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PROJECT_ROOT
from src.core.dataset.reader import DatasetReaderFactory
from src.logger import get_logger

logger = get_logger(__name__)

# Available datasets for pre-training
AVAILABLE_DATASETS = [
    "BBC_DB",
    "D_W_15K_V1",
    "D_W_15K_V2",
    "ICEWS_WIKI",
    "ICEWS_YAGO",
]

# Default configuration
DEFAULT_CONFIG = {
    "model_name": "google/flan-t5-xl",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "max_len_in": 128,
    "max_pairs_per_pred": 2000,
}


def get_dataset_path(dataset_name: str) -> Path:
    """Get the path to a dataset."""
    return PROJECT_ROOT / "data" / "raw" / "openea" / dataset_name


def load_dataset(dataset_name: str):
    """Load a dataset using the OpenEA reader."""
    dataset_path = get_dataset_path(dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    reader = DatasetReaderFactory.create_reader("openea")
    logger.info(f"Loading dataset {dataset_name} from {dataset_path}...")
    dataset = reader.read(str(dataset_path))

    logger.info(
        f"Dataset loaded: {len(dataset.aligned_entities)} aligned pairs, "
        f"{len(dataset.knowledge_graph_source)} source triples, "
        f"{len(dataset.knowledge_graph_target)} target triples"
    )

    return dataset




def pretrain_dataset(dataset_name: str, output_dir: Path, config: dict, dry_run: bool = False):
    """Pre-train FLAN-T5-XL on a single dataset."""
    logger.info("=" * 60)
    logger.info(f"PRE-TRAINING: {dataset_name}")
    logger.info("=" * 60)

    model_dir = output_dir / dataset_name
    if model_dir.exists() and (model_dir / "adapter_config.json").exists():
        logger.info(f"Model already exists at {model_dir}, skipping.")
        return True

    if dry_run:
        logger.info(f"[DRY-RUN] Would train model for {dataset_name}")
        logger.info(f"[DRY-RUN] Output: {model_dir}")
        return True

    # Load dataset
    try:
        dataset = load_dataset(dataset_name)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    # Initialize interpolator
    from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator

    logger.info("Initializing MixupT5XLInterpolator...")
    interpolator = MixupT5XLInterpolator(
        model_name=config["model_name"],
        out_dir=str(model_dir),
        device="cuda",
        max_len_in=config["max_len_in"],
    )

    # Build training data
    from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

    logger.info("Building training data...")
    builder = MixupDataBuilder()
    training_rows, canonical_mapping = builder.build_training_data(
        dataset=dataset,
        max_pairs_per_pred=config["max_pairs_per_pred"],
    )

    if not training_rows:
        logger.error(f"No training data generated for {dataset_name}")
        return False

    logger.info(f"Generated {len(training_rows)} training rows.")

    # Fine-tune
    logger.info("Starting fine-tuning...")
    interpolator.fine_tune(
        training_rows=training_rows,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
    )

    logger.info(f"Model saved to {model_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train FLAN-T5-XL model for each dataset."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=AVAILABLE_DATASETS,
        help="Datasets to pre-train on",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pre-train on all available datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "pretrained_plm",
        help="Output directory for pre-trained models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate",
    )

    args = parser.parse_args()

    if not args.datasets and not args.all:
        parser.error("Either --datasets or --all must be specified")

    datasets = AVAILABLE_DATASETS if args.all else args.datasets

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PLM PRE-TRAINING SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config: {config}")
    if args.dry_run:
        logger.info("MODE: DRY-RUN")
    logger.info("=" * 60)

    # Pre-train each dataset
    results = {}
    for dataset_name in datasets:
        success = pretrain_dataset(
            dataset_name=dataset_name,
            output_dir=args.output_dir,
            config=config,
            dry_run=args.dry_run,
        )
        results[dataset_name] = "OK" if success else "FAILED"

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for dataset_name, status in results.items():
        logger.info(f"  {dataset_name}: {status}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
