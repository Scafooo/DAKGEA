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
        ├── validation_report.txt  <- NEW: Score report
        └── ...
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdflib import Literal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PROJECT_ROOT
from src.core.dataset.reader import DatasetReaderFactory
from src.logger import get_logger

logger = get_logger(__name__)

# Validation config
SWEEP_SAMPLES = 50
REPORT_SAMPLES = 100

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


def collect_aligned_test_data(dataset, canonical_map, attr_map):
    """Collect aligned pairs for testing, grouped by predicate.

    Uses same predicate naming as training (clean_predicate with attr_map).
    """
    from src.augmentation.methods.plm.mixup_data_builder import clean_predicate

    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target

    # Collect literals per entity
    src_lits = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal):
            src_lits[s].append((p, str(o)))

    tgt_lits = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal):
            tgt_lits[s].append((p, str(o)))

    # Collect aligned pairs by predicate (same naming as training!)
    aligned_pool = defaultdict(list)
    for s_uri, t_uri in dataset.aligned_entities:
        s_attrs = src_lits.get(s_uri, [])
        t_attrs = tgt_lits.get(t_uri, [])

        for ps, vs in s_attrs:
            p_name = clean_predicate(ps, attr_map).replace('_', ' ')
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    vs_c, vt_c = vs.strip().lower(), vt.strip().lower()
                    if vs_c != vt_c:
                        aligned_pool[p_name].append((vs.strip(), vt.strip()))
                        break

    return aligned_pool


def collect_orphan_data(dataset, attr_map):
    """Collect orphan values grouped by predicate.

    Uses same predicate naming as training (clean_predicate with attr_map).
    """
    from src.augmentation.methods.plm.mixup_data_builder import clean_predicate

    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target

    orphans_by_pred = defaultdict(list)
    for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
        if isinstance(o, Literal):
            val = str(o).strip()
            if val:
                p_name = clean_predicate(p, attr_map).replace('_', ' ')
                orphans_by_pred[p_name].append(val)

    # Deduplicate
    return {p: list(set(vals)) for p, vals in orphans_by_pred.items()}


def validate_model(interpolator, dataset, canonical_map, report_path: Path, dataset_path: str = None):
    """Run validation sweep and generate report with scores."""
    from src.augmentation.methods.plm.scoring import calculate_pair_score, calculate_score
    from src.augmentation.methods.plm.mixup_data_builder import load_attr_names

    # Load attr_map for consistent predicate naming with training
    attr_map = {}
    if dataset_path:
        attr_map = load_attr_names(dataset_path)
        logger.info(f"Loaded {len(attr_map)} attribute name mappings for validation")

    logger.info("Collecting test data for validation...")
    aligned_pool = collect_aligned_test_data(dataset, canonical_map, attr_map)
    orphans_by_pred = collect_orphan_data(dataset, attr_map)

    # Build sweep pool
    sweep_pool = []
    all_preds = list(aligned_pool.keys())
    if all_preds:
        for _ in range(min(SWEEP_SAMPLES, sum(len(v) for v in aligned_pool.values()))):
            p = random.choice(all_preds)
            if aligned_pool[p]:
                v1, v2 = random.choice(aligned_pool[p])
                sweep_pool.append((p, v1, v2))

    if not sweep_pool:
        logger.warning("No aligned pairs available for validation")
        return None

    # Parameter sweep
    logger.info(f"Running parameter sweep on {len(sweep_pool)} samples...")
    sweep_results = []
    for alpha in [0.3, 0.4, 0.5]:
        for noise in [0.0, 0.01, 0.02]:
            for temp in [0.7]:
                interpolator.latent_noise_std = noise
                interpolator.gen_temperature = temp
                scores = []
                for p, v1, v2 in sweep_pool:
                    try:
                        aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=alpha)
                        res = calculate_pair_score(v1, v2, aa, ab)
                        scores.append(res["score"])
                    except Exception as e:
                        logger.warning(f"Interpolation error: {e}")
                        continue

                if scores:
                    avg = np.mean(scores)
                    sweep_results.append({"alpha": alpha, "noise": noise, "temp": temp, "score": avg})
                    logger.info(f"  Alpha={alpha} Noise={noise} Temp={temp} -> Score: {avg:.3f}")

    if not sweep_results:
        logger.warning("No valid sweep results")
        return None

    best = max(sweep_results, key=lambda x: x["score"])
    logger.info(f"BEST CONFIG: {best}")

    # Generate report
    logger.info("Generating validation report...")
    interpolator.latent_noise_std = best["noise"]
    interpolator.gen_temperature = best["temp"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("FLAN-T5-XL VALIDATION REPORT\n")
        f.write(f"Best config: alpha={best['alpha']}, noise={best['noise']}, temp={best['temp']}\n")
        f.write(f"Best score: {best['score']:.4f}\n")
        f.write("=" * 100 + "\n\n")

        # Section 1: Aligned pairs
        f.write("SECTION 1: ALIGNED PAIRS MIXUP\n")
        f.write("-" * 80 + "\n\n")

        p_names = list(aligned_pool.keys())
        count = 0
        while count < REPORT_SAMPLES // 2 and p_names:
            for p in list(p_names):
                if not aligned_pool[p]:
                    p_names.remove(p)
                    continue
                v1, v2 = aligned_pool[p].pop(random.randrange(len(aligned_pool[p])))
                try:
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best["alpha"])
                    res = calculate_pair_score(v1, v2, aa, ab)
                    f.write(f"PAIR {count+1:03d} | {p[:20]:20}\n")
                    f.write(f"  V1: {v1[:50]:50} -> AUG: {aa[:50]:50}\n")
                    f.write(f"  V2: {v2[:50]:50} -> AUG: {ab[:50]:50}\n")
                    f.write(f"  Score: {res['score']:.2f}\n")
                    f.write("-" * 80 + "\n")
                    count += 1
                except Exception as e:
                    f.write(f"PAIR {count+1:03d} | {p[:20]:20} | ERROR: {e}\n")
                    count += 1
                if count >= REPORT_SAMPLES // 2:
                    break

        # Section 2: Orphan variations
        f.write("\nSECTION 2: ORPHAN VARIATIONS\n")
        f.write("-" * 80 + "\n\n")

        o_names = list(orphans_by_pred.keys())
        o_count = 0
        while o_count < REPORT_SAMPLES // 2 and o_names:
            for p in list(o_names):
                if not orphans_by_pred[p]:
                    o_names.remove(p)
                    continue
                val = orphans_by_pred[p].pop(random.randrange(len(orphans_by_pred[p])))
                try:
                    aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
                    score = calculate_score(val, aa)
                    f.write(f"ORPHAN {o_count+1:03d} | {p[:20]:20}\n")
                    f.write(f"  ORIG: {val[:50]:50} -> VAR: {aa[:50]:50}\n")
                    f.write(f"  Score: {score:.2f}\n")
                    f.write("-" * 80 + "\n")
                    o_count += 1
                except Exception as e:
                    f.write(f"ORPHAN {o_count+1:03d} | {p[:20]:20} | ERROR: {e}\n")
                    o_count += 1
                if o_count >= REPORT_SAMPLES // 2:
                    break

        # Summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("SWEEP RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n")
        for res in sorted(sweep_results, key=lambda x: x["score"], reverse=True):
            marker = " <- BEST" if res == best else ""
            f.write(f"  alpha={res['alpha']:.1f}, noise={res['noise']:.2f}, temp={res['temp']:.1f} -> {res['score']:.4f}{marker}\n")
        f.write("=" * 100 + "\n")

    logger.info(f"Validation report saved to {report_path}")
    return best




def pretrain_dataset(dataset_name: str, output_dir: Path, config: dict, dry_run: bool = False, skip_validation: bool = False):
    """Pre-train FLAN-T5-XL on a single dataset."""
    logger.info("=" * 60)
    logger.info(f"PRE-TRAINING: {dataset_name}")
    logger.info("=" * 60)

    model_dir = output_dir / dataset_name
    if model_dir.exists() and (model_dir / "adapter_config.json").exists():
        logger.info(f"Model already exists at {model_dir}, skipping training.")
        # Still run validation if report doesn't exist
        if not skip_validation and not (model_dir / "validation_report.txt").exists():
            logger.info("Running validation for existing model...")
            try:
                dataset = load_dataset(dataset_name)
                from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator
                from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

                interpolator = MixupT5XLInterpolator(
                    model_name=config["model_name"],
                    out_dir=str(model_dir),
                    device="cuda",
                    max_len_in=config["max_len_in"],
                    pretrained_path=str(model_dir),
                )
                builder = MixupDataBuilder()
                dataset_path_str = str(get_dataset_path(dataset_name))
                _, canonical_mapping = builder.build_training_data(dataset, max_pairs_per_pred=10, dataset_path=dataset_path_str)
                validate_model(interpolator, dataset, canonical_mapping, model_dir / "validation_report.txt", dataset_path=dataset_path_str)
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
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
        dataset_path=str(get_dataset_path(dataset_name)),
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

    # Post-training validation
    if not skip_validation:
        logger.info("Running post-training validation...")
        try:
            best_config = validate_model(
                interpolator, dataset, canonical_mapping, model_dir / "validation_report.txt",
                dataset_path=str(get_dataset_path(dataset_name))
            )
            if best_config:
                logger.info(f"Best generation config: {best_config}")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")

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
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-training validation and report generation",
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
            skip_validation=args.skip_validation,
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
