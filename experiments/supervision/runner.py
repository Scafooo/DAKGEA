"""Runner for supervision level experiments.

This script orchestrates experiments at multiple supervision levels (r%) to measure
the effectiveness of data augmentation when training data is limited.

Experimental Setup:
1. Initial Split: M_gold -> M_pool (20%) + M_test (80%)
2. For each r% in [10%, 20%, ..., 100%]:
   a. Sample M_train^(r) = r% of M_pool
   b. BASELINE: Train model on original KGs + M_train^(r)
   c. AUGMENTED: Apply augmentation to create M_aug^(r) + G'
   d. Train model on augmented KGs + M_train^(r) + M_aug^(r)
   e. Evaluate both on FIXED M_test
3. Output: Table comparing baseline vs augmented at each r level

Key Principle: M_test is the SAME for all r levels (apples-to-apples comparison).
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.alignment_models.registry import get_alignment_model, load_builtin_models
from src.augmentation.registry import AUGMENTATION_REGISTRY, load_builtin_augmentations
from src.config.loader import PROJECT_ROOT
from src.core.dataset import Dataset
from src.core.dataset.reader import DatasetReaderFactory
from src.logger import get_logger

from .config import SupervisionExperimentConfig
from .splitter import SupervisionExperimentSplitter, SupervisionSplit, SupervisionLevelData
from .writer import SupervisionExperimentWriter

logger = get_logger(__name__)


class SupervisionExperimentRunner:
    """Runs supervision level experiments with baseline vs augmented comparison."""

    def __init__(self, config: SupervisionExperimentConfig):
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.splitter = SupervisionExperimentSplitter(seed=config.seed)
        self.output_dir = Path(config.output_dir)
        self.results: Dict[float, Dict[str, Any]] = {}

        # Initialize registries
        load_builtin_augmentations()
        load_builtin_models()

        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[float, Dict[str, Any]]:
        """Run the complete supervision experiment.

        Returns:
            Dictionary mapping supervision level -> results
        """
        logger.info(f"=" * 60)
        logger.info(f"SUPERVISION LEVEL EXPERIMENT: {self.config.name}")
        logger.info(f"=" * 60)
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Levels: {[f'{l:.0%}' for l in self.config.levels]}")
        logger.info(f"Augmentation: {self.config.augmentation_method} (ratio={self.config.augmentation_ratio})")
        logger.info(f"Models: {self.config.models}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"=" * 60)

        # Step 1: Load dataset
        dataset = self._load_dataset()
        logger.info(
            f"Loaded dataset: {len(dataset.aligned_entities)} aligned pairs, "
            f"{len(dataset.knowledge_graph_source)} source triples, "
            f"{len(dataset.knowledge_graph_target)} target triples"
        )

        # Step 2: Create M_pool/M_test split
        split_cache = self.output_dir / "split.json"
        split = self.splitter.split_pool_test(
            dataset,
            pool_ratio=self.config.pool_ratio,
            cache_path=split_cache,
        )
        logger.info(
            f"Split: M_pool={split.pool_size} ({self.config.pool_ratio:.0%}), "
            f"M_test={split.test_size} ({1-self.config.pool_ratio:.0%})"
        )

        # Step 3: Run experiment at each supervision level
        for level in self.config.levels:
            logger.info(f"\n{'='*40}")
            logger.info(f"SUPERVISION LEVEL: {level:.0%}")
            logger.info(f"{'='*40}")

            try:
                level_results = self._run_level(dataset, split, level)
                self.results[level] = level_results
                self._save_level_results(level, level_results)
            except Exception as e:
                logger.error(f"Error at level {level:.0%}: {e}")
                self.results[level] = {"error": str(e)}

        # Step 4: Generate summary
        self._generate_summary()

        return self.results

    def _load_dataset(self) -> Dataset:
        """Load the dataset based on configuration."""
        dataset_name = self.config.dataset_name

        # Determine dataset path
        if "/" in dataset_name:
            # Path like "openea/D_W_15K_V1"
            dataset_path = PROJECT_ROOT / "data" / "raw" / dataset_name
        else:
            dataset_path = Path(dataset_name)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Auto-detect reader type
        reader_type = self._detect_reader_type(dataset_path)
        reader = DatasetReaderFactory.create_reader(reader_type)

        logger.info(f"Loading dataset from {dataset_path} using {reader_type} reader")
        return reader.read(str(dataset_path))

    def _detect_reader_type(self, path: Path) -> str:
        """Detect the appropriate reader type for a dataset path."""
        if (path / "attribute_data").exists() or (path / "knowformer_data").exists():
            return "openea"
        elif (path / "ent_ids_1").exists():
            return "bert_int"
        else:
            return "openea"  # Default

    def _run_level(
        self,
        original_dataset: Dataset,
        split: SupervisionSplit,
        level: float,
    ) -> Dict[str, Any]:
        """Run experiment at a single supervision level.

        Returns:
            Dictionary with baseline and augmented results
        """
        level_tag = self.config.get_level_tag(level)
        level_dir = self.config.get_level_dir(level)
        level_dir.mkdir(parents=True, exist_ok=True)

        # Sample training pairs for this level
        level_data = self.splitter.sample_supervision_level(split, level)

        # Create training dataset (graphs intact, reduced alignments)
        train_dataset, test_pairs = self.splitter.create_level_dataset(
            original_dataset, level_data
        )

        logger.info(
            f"Level {level:.0%}: train={level_data.train_size}, "
            f"test={level_data.test_size} (fixed)"
        )

        results = {
            "level": level,
            "level_tag": level_tag,
            "train_size": level_data.train_size,
            "test_size": level_data.test_size,
            "baseline": {},
            "augmented": {},
        }

        # Create writer with fixed test set
        writer = SupervisionExperimentWriter(
            fixed_test_pairs=test_pairs,
            validation_ratio=0.1,
            augmented_in_train_only=True,
        )

        # --- BASELINE ---
        baseline_dir = level_dir / "baseline"
        if self._check_cached_results(baseline_dir) and self.config.resume:
            logger.info(f"[BASELINE] Loading cached results from {baseline_dir}")
            results["baseline"] = self._load_cached_results(baseline_dir)
        else:
            logger.info(f"[BASELINE] Running at level {level:.0%}")
            baseline_results = self._run_variant(
                "baseline",
                train_dataset,
                test_pairs,
                writer,
                baseline_dir,
                augment=False,
            )
            results["baseline"] = baseline_results

        # --- AUGMENTED ---
        augmented_dir = level_dir / "augmented"
        if self._check_cached_results(augmented_dir) and self.config.resume:
            logger.info(f"[AUGMENTED] Loading cached results from {augmented_dir}")
            results["augmented"] = self._load_cached_results(augmented_dir)
        else:
            logger.info(f"[AUGMENTED] Running at level {level:.0%}")
            augmented_results = self._run_variant(
                "augmented",
                train_dataset,
                test_pairs,
                writer,
                augmented_dir,
                augment=True,
            )
            results["augmented"] = augmented_results

        # Log comparison
        self._log_comparison(level, results)

        return results

    def _run_variant(
        self,
        variant_name: str,
        train_dataset: Dataset,
        test_pairs,
        writer: SupervisionExperimentWriter,
        output_dir: Path,
        augment: bool,
    ) -> Dict[str, Any]:
        """Run a single variant (baseline or augmented).

        Args:
            variant_name: "baseline" or "augmented"
            train_dataset: Dataset with training alignments
            test_pairs: Fixed test set
            writer: Writer configured with fixed test set
            output_dir: Output directory for this variant
            augment: Whether to apply augmentation

        Returns:
            Results dictionary with metrics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "dataset" / self.config.writer

        # Clone dataset for this variant
        variant_dataset = train_dataset.clone()
        original_pairs = len(variant_dataset.aligned_entities)

        # Apply augmentation if requested
        if augment:
            variant_dataset = self._apply_augmentation(variant_dataset, output_dir)
            augmented_pairs = len(variant_dataset.aligned_entities)
            synthetic_pairs = augmented_pairs - original_pairs
            logger.info(
                f"[{variant_name.upper()}] After augmentation: "
                f"{augmented_pairs} pairs ({original_pairs} original + {synthetic_pairs} synthetic)"
            )
        else:
            augmented_pairs = original_pairs
            synthetic_pairs = 0

        # Write dataset
        logger.info(f"[{variant_name.upper()}] Writing dataset to {dataset_dir}")
        writer.write(variant_dataset, str(dataset_dir))

        # Evaluate models
        model_results = {}
        for model_name in self.config.models:
            logger.info(f"[{variant_name.upper()}] Evaluating model: {model_name}")
            metrics = self._evaluate_model(model_name, dataset_dir)
            model_results[model_name] = metrics

        # Save results
        results = {
            "variant": variant_name,
            "original_pairs": original_pairs,
            "augmented_pairs": augmented_pairs,
            "synthetic_pairs": synthetic_pairs,
            "models": model_results,
        }

        results_file = output_dir / "results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def _apply_augmentation(self, dataset: Dataset, output_dir: Path) -> Dataset:
        """Apply augmentation to dataset."""
        augmenter_cls = AUGMENTATION_REGISTRY.get(self.config.augmentation_method)

        # Build augmentation config
        aug_config = {
            "augmentation": {
                "method": self.config.augmentation_method,
                "ratio": self.config.augmentation_ratio,
                "stage_root": str(output_dir),
                **self.config.augmentation_config,
            },
            "experiment": {
                "seed": self.config.seed,
            },
        }

        augmenter = augmenter_cls(aug_config)
        return augmenter.augment(dataset)

    def _evaluate_model(self, model_name: str, dataset_dir: Path) -> Dict[str, float]:
        """Evaluate a model on the dataset.

        Returns:
            Dictionary with metrics (hits@1, hits@5, hits@10, mrr)
        """
        model_cls = get_alignment_model(model_name)

        # Build evaluation config
        eval_config = {
            "lineage": {
                "dataset_workspace": str(dataset_dir),
            },
            "experiment": {
                "seed": self.config.seed,
            },
        }

        model = model_cls(eval_config)

        # Evaluate returns metrics dict
        # Note: We need to pass datasets but model reads from dataset_workspace
        # Create dummy datasets since model loads from disk
        dummy_dataset = Dataset(
            knowledge_graph_source=None,
            knowledge_graph_target=None,
            aligned_entities=[],
        )

        try:
            results = model.evaluate(dummy_dataset, dummy_dataset)
            return results
        except Exception as e:
            logger.error(f"Evaluation error for {model_name}: {e}")
            return {"error": str(e)}

    def _check_cached_results(self, output_dir: Path) -> bool:
        """Check if cached results exist."""
        results_file = output_dir / "results.json"
        return results_file.exists()

    def _load_cached_results(self, output_dir: Path) -> Dict[str, Any]:
        """Load cached results from disk."""
        results_file = output_dir / "results.json"
        with results_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_level_results(self, level: float, results: Dict[str, Any]) -> None:
        """Save results for a single level."""
        level_dir = self.config.get_level_dir(level)
        results_file = level_dir / "level_results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _log_comparison(self, level: float, results: Dict[str, Any]) -> None:
        """Log comparison between baseline and augmented."""
        baseline = results.get("baseline", {})
        augmented = results.get("augmented", {})

        logger.info(f"\n--- Level {level:.0%} Comparison ---")
        logger.info(f"Train size: {results['train_size']} | Test size: {results['test_size']} (fixed)")

        for model in self.config.models:
            baseline_metrics = baseline.get("models", {}).get(model, {})
            augmented_metrics = augmented.get("models", {}).get(model, {})

            b_h1 = baseline_metrics.get("hits@1", 0)
            a_h1 = augmented_metrics.get("hits@1", 0)
            diff = a_h1 - b_h1

            logger.info(
                f"  {model}: Baseline H@1={b_h1:.4f}, Augmented H@1={a_h1:.4f} "
                f"(Δ={diff:+.4f})"
            )

    def _generate_summary(self) -> None:
        """Generate summary table and save to file."""
        summary = {
            "experiment": self.config.name,
            "dataset": self.config.dataset_name,
            "augmentation": self.config.augmentation_method,
            "augmentation_ratio": self.config.augmentation_ratio,
            "pool_ratio": self.config.pool_ratio,
            "seed": self.config.seed,
            "timestamp": datetime.now().isoformat(),
            "levels": {},
        }

        # Build summary table
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY: Supervision Level Experiment")
        logger.info("=" * 80)

        header = f"{'Level':>8} | {'Train':>6} | {'Test':>6} |"
        for model in self.config.models:
            header += f" {model} B/A |"
        logger.info(header)
        logger.info("-" * len(header))

        for level in sorted(self.results.keys()):
            result = self.results[level]
            if "error" in result:
                logger.info(f"{level:>7.0%} | ERROR: {result['error']}")
                continue

            row = f"{level:>7.0%} | {result['train_size']:>6} | {result['test_size']:>6} |"

            level_summary = {
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "models": {},
            }

            for model in self.config.models:
                baseline_h1 = result.get("baseline", {}).get("models", {}).get(model, {}).get("hits@1", 0)
                augmented_h1 = result.get("augmented", {}).get("models", {}).get(model, {}).get("hits@1", 0)
                row += f" {baseline_h1:.3f}/{augmented_h1:.3f} |"

                level_summary["models"][model] = {
                    "baseline": result.get("baseline", {}).get("models", {}).get(model, {}),
                    "augmented": result.get("augmented", {}).get("models", {}).get(model, {}),
                }

            summary["levels"][f"{level:.0%}"] = level_summary
            logger.info(row)

        logger.info("=" * 80)

        # Save summary
        summary_file = self.output_dir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run supervision level experiments"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from cached results",
    )
    args = parser.parse_args()

    config = SupervisionExperimentConfig.from_yaml(args.config)
    if args.no_resume:
        config.resume = False

    runner = SupervisionExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
