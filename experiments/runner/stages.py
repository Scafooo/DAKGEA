"""Building blocks for the reduction/augmentation/evaluation stages."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.alignment_models.registry import MODEL_REGISTRY, get_alignment_model
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset
from src.logger import get_logger
from src.reduction.registry import REDUCTION_REGISTRY

from .specs import WriterPlan

logger = get_logger(__name__)

# Constants
RATIO_THRESHOLD = 0.999999
VARIANT_BASELINE = "baseline"
VARIANT_REDUCED = "reduced"
MODEL_BERT_INT = "bert_int"
WRITER_HYBEA = "hybea"
ATTRIBUTE_DATA_DIR = "attribute_data"


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_hybea_plan(plan: WriterPlan) -> bool:
    """Check if a WriterPlan corresponds to the hybea format."""
    return getattr(plan.writer, "file_type", None) == WRITER_HYBEA or plan.name == WRITER_HYBEA


def _is_baseline_variant(name: Optional[str]) -> bool:
    """Check if an augmentation name represents the baseline variant."""
    return name in (None, VARIANT_BASELINE)


class StageSummaryWriter:
    """Utility that writes stage summaries to JSON files."""

    @staticmethod
    def write(path: Path, payload: Dict[str, Any]) -> None:
        _ensure_directory(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


class ReductionStage:
    """Encapsulate the reduction logic and caching behaviour."""

    def __init__(
        self,
        reducer_name: str,
        writer_plans: Iterable[WriterPlan],
        resume: bool,
    ) -> None:
        self.reducer_cls = REDUCTION_REGISTRY.get(reducer_name)
        self.reducer_name = reducer_name
        self.writer_plans = list(writer_plans)
        self.resume = resume

    def execute(
        self,
        stage_cfg: Dict[str, Any],
        dataset: Dataset,
        reader,
        ratio: float,
        ratio_tag: str,
        lineage: Dict[str, Any],
        ratio_root: Path,
        ratio_meta: Dict[str, Any],
        subtype: Optional[str] = None,
    ) -> Dataset:
        reduction_root = Path(lineage.get("reduction_root", ratio_root / "reduction"))
        _ensure_directory(reduction_root)
        reader_plan = lineage.setdefault("reader_plan", {}).get("reduction")
        if reader_plan is None:
            reader_plan = self._select_reader_plan()
            lineage.setdefault("reader_plan", {})["reduction"] = reader_plan
        reader_root = reduction_root / reader_plan

        reduction_meta = ratio_meta.setdefault(
            "reduction", {"method": self.reducer_name, "paths": {}}
        )
        reduction_meta["method"] = self.reducer_name
        reduction_meta["reader_plan"] = reader_plan
        reduction_paths = reduction_meta.setdefault("paths", {})
        reduction_meta["summary"] = str((reduction_root / "summary.json").resolve())
        reduced_paths = lineage.setdefault("reduced_paths", {})
        reduced_datasets = lineage.setdefault("reduced_datasets", {})

        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping reduction — cached artefacts detected (ratio=%s)", ratio_tag
            )
            self._record_plan_paths(
                reduction_root,
                reduced_paths,
                reduced_datasets,
                reduction_paths,
                reduction_meta,
                lineage,
            )
            cached = self._read_cached_dataset(reader, reader_root, subtype)
            if cached:
                return cached
            logger.warning(
                "Cached reduction at %s incomplete; recomputing.", reader_root
            )

        reducer = self.reducer_cls(stage_cfg)
        dataset_reduced = reducer.reduce(dataset.clone())
        logger.info(
            "[SUCCESS] Reduction complete (%d aligned pairs)",
            len(dataset_reduced.aligned_entities),
        )

        for plan in self.writer_plans:
            plan_root = reduction_root / plan.name
            if plan.write_reduced:
                _ensure_directory(plan_root)
                plan.writer.write(dataset_reduced, str(plan_root))
                logger.info("📝 [%s] Saved reduced dataset → %s", plan.name, plan_root)
            self._maybe_record_plan(plan, plan_root, reduced_paths, reduced_datasets, reduction_paths, reduction_meta, lineage)

        StageSummaryWriter.write(
            reduction_root / "summary.json",
            {
                "method": self.reducer_name,
                "ratio": ratio,
                "target_entities": stage_cfg.get("reduction", {}).get("target_entities"),
                "aligned_pairs": len(dataset_reduced.aligned_entities),
                "writers": sorted(plan.name for plan in self.writer_plans if plan.write_reduced),
            },
        )
        return dataset_reduced

    @staticmethod
    def _has_cached(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    @staticmethod
    def _read_cached_dataset(reader, reader_root: Path, subtype: Optional[str]):
        try:
            return reader.read(str(reader_root), subtype=subtype)
        except FileNotFoundError:
            return None

    def _select_reader_plan(self) -> str:
        return self.writer_plans[0].name if self.writer_plans else "default"

    def _record_plan_paths(
        self,
        base_root: Path,
        reduced_paths: Dict[str, str],
        reduced_datasets: Dict[str, str],
        reduction_paths: Dict[str, str],
        reduction_meta: Dict[str, Any],
        lineage: Dict[str, Any],
    ) -> None:
        for plan in self.writer_plans:
            plan_root = base_root / plan.name
            if plan_root.exists():
                self._record_single_plan(plan, plan_root, reduced_paths, reduced_datasets,
                                        reduction_paths, reduction_meta, lineage)

    @staticmethod
    def _record_single_plan(
        plan: WriterPlan,
        plan_root: Path,
        reduced_paths: Dict[str, str],
        reduced_datasets: Dict[str, str],
        reduction_paths: Dict[str, str],
        reduction_meta: Dict[str, Any],
        lineage: Dict[str, Any],
    ) -> None:
        """Record paths for a single writer plan in lineage tracking."""
        resolved_path = str(plan_root.resolve())
        reduced_paths[plan.name] = resolved_path
        reduction_paths[plan.name] = resolved_path
        reduced_datasets[plan.name] = resolved_path
        if _is_hybea_plan(plan):
            lineage["reduced_hybea_path"] = resolved_path
            reduction_meta["hybea_path"] = resolved_path

    @staticmethod
    def _maybe_record_plan(plan, plan_root, reduced_paths, reduced_datasets, reduction_paths, reduction_meta, lineage):
        if plan_root.exists():
            ReductionStage._record_single_plan(plan, plan_root, reduced_paths, reduced_datasets,
                                               reduction_paths, reduction_meta, lineage)


class AugmentationStage:
    """Encapsulate augmentation logic and artefact management."""

    def __init__(
        self,
        writer_plans: Iterable[WriterPlan],
        resume: bool,
    ) -> None:
        self.writer_plans = list(writer_plans)
        self.resume = resume

    def execute(
        self,
        stage_cfg: Dict[str, Any],
        augmentation_name: str,
        dataset_reduced: Dataset,
        reader,
        lineage: Dict[str, Any],
        ratio: float,
        ratio_tag: str,
        ratio_root: Path,
        ratio_meta: Dict[str, Any],
        subtype: Optional[str] = None,
    ) -> Dataset:
        augmentation_root = Path(lineage.get("augmentation_root", ratio_root / "augmentation"))
        stage_root = augmentation_root / augmentation_name
        _ensure_directory(stage_root)
        reader_plan = lineage.setdefault("reader_plan", {}).get("augmentation")
        if reader_plan is None:
            reader_plan = self._select_reader_plan(reader)
            lineage.setdefault("reader_plan", {})["augmentation"] = reader_plan
        reader_root = stage_root / reader_plan

        augmentations_meta = ratio_meta.setdefault("augmentations", {})
        augmentation_meta = augmentations_meta.setdefault(
            augmentation_name, {"paths": {}, "reader_plan": reader_plan}
        )
        augmentation_meta["reader_plan"] = reader_plan
        augmentation_meta["summary"] = str((stage_root / "summary.json").resolve())
        augmentation_paths = augmentation_meta.setdefault("paths", {})

        augmented_paths = lineage.setdefault("augmented_paths", {})
        per_aug_paths = augmented_paths.setdefault(augmentation_name, {})
        augmented_datasets = lineage.setdefault("augmented_datasets", {})
        per_aug_datasets = augmented_datasets.setdefault(augmentation_name, {})
        hybea_aug_paths = lineage.setdefault("augmented_hybea_paths", {})
        lineage.setdefault("augmentation_roots", {})[
            augmentation_name
        ] = str(stage_root.resolve())

        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping augmentation '%s' — cached artefacts detected (ratio=%s)",
                augmentation_name,
                ratio_tag,
            )
            for plan in self.writer_plans:
                plan_root = stage_root / plan.name
                if plan_root.exists():
                    resolved_path = str(plan_root.resolve())
                    per_aug_paths[plan.name] = resolved_path
                    augmentation_paths[plan.name] = resolved_path
                    per_aug_datasets[plan.name] = resolved_path
                    if _is_hybea_plan(plan):
                        hybea_aug_paths[augmentation_name] = resolved_path
                        augmentation_meta["hybea_path"] = resolved_path
            cached = ReductionStage._read_cached_dataset(reader, reader_root, subtype)
            if cached:
                return cached
            logger.warning(
                "Cached augmentation '%s' incomplete at %s; recomputing.",
                augmentation_name,
                reader_root,
            )

        augmenter_cls = AUGMENTATION_REGISTRY.get(augmentation_name)
        augmenter = augmenter_cls(stage_cfg)
        dataset_augmented = augmenter.augment(dataset_reduced.clone())
        logger.info(
            "[SUCCESS] Augmentation '%s' complete (%d aligned pairs)",
            augmentation_name,
            len(dataset_augmented.aligned_entities),
        )

        for plan in self.writer_plans:
            plan_root = stage_root / plan.name
            if plan.write_augmented:
                _ensure_directory(plan_root)
                plan.writer.write(dataset_augmented, str(plan_root))
                logger.info("📝 [%s] Saved augmented dataset → %s", plan.name, plan_root)
            if plan_root.exists():
                resolved_path = str(plan_root.resolve())
                per_aug_paths[plan.name] = resolved_path
                augmentation_paths[plan.name] = resolved_path
                per_aug_datasets[plan.name] = resolved_path
                if _is_hybea_plan(plan):
                    hybea_aug_paths[augmentation_name] = resolved_path
                    augmentation_meta["hybea_path"] = resolved_path

        StageSummaryWriter.write(
            stage_root / "summary.json",
            {
                "augmentation": augmentation_name,
                "ratio": ratio,
                "aligned_pairs": len(dataset_augmented.aligned_entities),
                "writers": sorted(plan.name for plan in self.writer_plans if plan.write_augmented),
            },
        )
        return dataset_augmented

    @staticmethod
    def _has_cached(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    @staticmethod
    def _select_reader_plan(reader) -> str:
        return getattr(reader, "file_type", "default")


class EvaluationStage:
    """Encapsulate evaluation logic and result handling."""

    def __init__(
        self,
        writer_plans: Iterable[WriterPlan],
        models: Iterable[str],
        resume: bool,
    ) -> None:
        self.writer_plans = list(writer_plans)
        self.models = list(models)
        self.resume = resume

    def execute(
        self,
        augmentation_name: str,
        dataset_reduced: Dataset,
        dataset_augmented: Dataset,
        stage_cfg: Dict[str, Any],
        lineage_cfg: Dict[str, Any],
        ratio_root: Path,
        ratio_tag: str,
        ratio_meta: Dict[str, Any],
    ) -> None:
        evaluation_dir = self._resolve_evaluation_dir(lineage_cfg, ratio_root, augmentation_name)
        variant_key = self._normalise_variant_key(augmentation_name)

        evaluations_meta = ratio_meta.setdefault("evaluations", {})
        evaluation_meta = evaluations_meta.setdefault(variant_key, {})
        evaluation_meta["summary"] = str((evaluation_dir / "summary.json").resolve())
        evaluation_paths = evaluation_meta.setdefault("paths", {})

        base_root = Path(lineage_cfg.get("evaluation_root", ratio_root / "evaluation"))

        for model_name in self.models:
            out_file = evaluation_dir / f"{model_name}.json"
            if self.resume and not out_file.exists():
                self._migrate_legacy_result(base_root, variant_key, out_file)
            evaluation_paths[model_name] = str(out_file.resolve())
            if self.resume and out_file.exists():
                logger.info(
                    "⏭️  Skipping model '%s' (%s) — results already cached",
                    model_name,
                    augmentation_name,
                )
                continue

            stage_cfg_eval = copy.deepcopy(stage_cfg)
            experiment_meta = stage_cfg_eval.setdefault("experiment", {})
            experiment_meta["augmentation"] = augmentation_name
            experiment_meta["reduction_method"] = stage_cfg.get("experiment", {}).get("reduction_method")

            lineage = stage_cfg_eval.setdefault("lineage", {})
            lineage.setdefault("reduction_method", stage_cfg.get("lineage", {}).get("reduction_method"))
            lineage["augmentation_name"] = (
                augmentation_name if not _is_baseline_variant(augmentation_name) else None
            )
            lineage["active_source"] = (
                "augmented" if not _is_baseline_variant(augmentation_name) else "reduced"
            )

            if _is_baseline_variant(augmentation_name):
                hybea_path = lineage.get("reduced_hybea_path")
                hybea_base = lineage_cfg.get("reduced_base")
            else:
                hybea_path = lineage.get("augmented_hybea_paths", {}).get(augmentation_name)
                hybea_base = lineage_cfg.get("augmented_base")

            lineage["hybea_dataset_path"] = hybea_path
            lineage["hybea_dataset_base"] = hybea_base

            overrides = self._build_model_overrides(
                model_name,
                augmentation_name,
                stage_cfg_eval,
                lineage_cfg,
                evaluation_dir,
                variant_key,
            )
            if overrides:
                if len(self.models) == 1:
                    stage_cfg_eval.setdefault("model", {}).update(overrides)
                else:
                    stage_cfg_eval.setdefault("models", {}).setdefault(model_name, {}).update(overrides)

            model_cls = get_alignment_model(model_name)
            logger.info("[STEP] → Evaluating model '%s' (augmentation=%s)", model_name, augmentation_name)
            model = model_cls(stage_cfg_eval)
            results = model.evaluate(dataset_reduced, dataset_augmented)
            logger.info("[SUCCESS] Model '%s' evaluation finished", model_name)

            self._write_results(out_file, results)

        StageSummaryWriter.write(
            evaluation_dir / "summary.json",
            {
                "augmentation": augmentation_name if not _is_baseline_variant(augmentation_name) else None,
                "variant": variant_key,
                "models": sorted(self.models),
                "files": sorted(evaluation_paths.values()),
            },
        )

    @staticmethod
    def _resolve_evaluation_dir(
        lineage_cfg: Dict[str, Any],
        ratio_root: Path,
        augmentation_name: str,
    ) -> Path:
        evaluation_root_str = lineage_cfg.get("evaluation_root")
        evaluation_root = Path(evaluation_root_str) if evaluation_root_str else ratio_root / "evaluation"
        key = EvaluationStage._normalise_variant_key(augmentation_name)
        target = evaluation_root / key
        _ensure_directory(target)
        lineage_cfg.setdefault("evaluation_dirs", {})[key] = str(target.resolve())
        return target

    @staticmethod
    def _normalise_variant_key(name: Optional[str]) -> str:
        if _is_baseline_variant(name):
            return VARIANT_REDUCED
        return name.replace("/", "_")

    @staticmethod
    def _migrate_legacy_result(base_root: Path, variant_key: str, destination: Path) -> None:
        ratio_root = destination.parent.parent.parent
        candidates = [
            base_root / "results" / destination.name,
            ratio_root / "results" / variant_key / destination.name,
        ]
        for legacy_file in candidates:
            if legacy_file.exists() and not destination.exists():
                _ensure_directory(destination.parent)
                try:
                    legacy_file.replace(destination)
                except OSError:
                    shutil.copy2(legacy_file, destination)
                    legacy_file.unlink(missing_ok=True)
                break

    @staticmethod
    def _select_primary_path(path_mapping: Optional[Dict[str, str]]) -> Optional[Path]:
        """Select the primary path from a mapping, preferring 'hybea' if available."""
        if not path_mapping:
            return None
        if WRITER_HYBEA in path_mapping and path_mapping[WRITER_HYBEA]:
            return Path(path_mapping[WRITER_HYBEA])
        for value in path_mapping.values():
            if value:
                return Path(value)
        return None

    def _select_dataset_root_for_model(
        self,
        augmentation_name: Optional[str],
        lineage_cfg: Dict[str, Any],
        variant_key: str,
    ) -> Optional[Path]:
        """
        Select the appropriate dataset root based on augmentation variant.

        For baseline variants, selects from reduced datasets. For augmented variants,
        selects from the augmented datasets. Falls back to raw path if reduction
        was not executed.

        Args:
            augmentation_name: Name of augmentation method, or None/"baseline" for baseline
            lineage_cfg: Lineage configuration containing dataset paths
            variant_key: Normalized variant key for error messages

        Returns:
            Path: Selected dataset root path

        Raises:
            ValueError: If no suitable dataset path is found
        """
        reduced_map = lineage_cfg.get("reduced_datasets", {})
        augmented_map = lineage_cfg.get("augmented_datasets", {}).get(augmentation_name, {})

        if _is_baseline_variant(augmentation_name):
            dataset_root = self._select_primary_path(reduced_map)
        else:
            dataset_root = self._select_primary_path(augmented_map)

        # Fallback to raw path if reduction was not executed
        if dataset_root is None:
            reduction_executed = lineage_cfg.get("_reduction_executed", False)
            raw_source = lineage_cfg.get("raw_source")
            raw_path = Path(raw_source) if raw_source else None

            if not reduction_executed and raw_path is not None:
                return raw_path
            else:
                raise ValueError(
                    f"No dataset artefacts available for variant '{variant_key}'. "
                    f"reduced_map={reduced_map}, augmented_map={augmented_map}, "
                    f"reduction_executed={reduction_executed}, raw_path={raw_path}"
                )

        return dataset_root

    @staticmethod
    def _resolve_attribute_directory(
        dataset_root: Path,
        raw_path: Optional[Path],
        reduction_executed: bool,
        variant_key: str,
    ) -> Path:
        """
        Find the attribute_data directory in the dataset hierarchy.

        Searches for the 'attribute_data' subdirectory in multiple locations:
        1. Direct child of dataset_root
        2. Under dataset_root/hybea/
        3. In raw_path (if reduction was not executed)

        Args:
            dataset_root: Base dataset directory to search
            raw_path: Original raw dataset path (fallback)
            reduction_executed: Whether reduction stage was executed
            variant_key: Variant key for error messages

        Returns:
            Path: Resolved path to attribute_data directory

        Raises:
            ValueError: If attribute_data directory cannot be found
        """
        # Try direct attribute_data subdirectory
        attribute_dir = dataset_root / ATTRIBUTE_DATA_DIR
        if attribute_dir.exists():
            return attribute_dir

        # Try hybea/attribute_data subdirectory
        hybea_dir = dataset_root / WRITER_HYBEA / ATTRIBUTE_DATA_DIR
        if hybea_dir.exists():
            return hybea_dir

        # Fallback to raw path if reduction was not executed
        if raw_path is not None and not reduction_executed:
            raw_attr = raw_path / ATTRIBUTE_DATA_DIR
            return raw_attr if raw_attr.exists() else raw_path

        raise ValueError(
            f"Dataset path '{dataset_root}' is missing required '{ATTRIBUTE_DATA_DIR}' "
            f"directory for variant '{variant_key}'."
        )

    @staticmethod
    def _create_model_save_directory(
        lineage_cfg: Dict[str, Any],
        evaluation_dir: Path,
        variant_key: str,
    ) -> Path:
        """Create and return the model checkpoint save directory."""
        save_root = Path(lineage_cfg.get("evaluation_root", evaluation_dir.parent)) / MODEL_BERT_INT
        save_variant_dir = save_root / variant_key
        save_variant_dir.mkdir(parents=True, exist_ok=True)
        return save_variant_dir

    def _build_model_overrides(
        self,
        model_name: str,
        augmentation_name: Optional[str],
        stage_cfg_eval: Dict[str, Any],
        lineage_cfg: Dict[str, Any],
        evaluation_dir: Path,
        variant_key: str,
    ) -> Dict[str, Any]:
        """Build model-specific configuration overrides for evaluation."""
        if model_name != MODEL_BERT_INT:
            return {}

        # Select the appropriate dataset root
        dataset_root = self._select_dataset_root_for_model(
            augmentation_name, lineage_cfg, variant_key
        )

        # Find the attribute_data directory
        raw_source = lineage_cfg.get("raw_source")
        raw_path = Path(raw_source) if raw_source else None
        reduction_executed = lineage_cfg.get("_reduction_executed", False)

        dataset_root = self._resolve_attribute_directory(
            dataset_root, raw_path, reduction_executed, variant_key
        )

        # Create model save directory
        save_variant_dir = self._create_model_save_directory(
            lineage_cfg, evaluation_dir, variant_key
        )

        # Construct overrides
        overrides: Dict[str, Any] = {
            "paths": {
                "dataset_root": str(dataset_root.resolve()),
                "model_save_dir": str(save_variant_dir.resolve()),
                "model_save_prefix": variant_key,
            }
        }

        # Add dataset name if available
        experiment_meta = stage_cfg_eval.get("experiment", {})
        dataset_name = experiment_meta.get("dataset")
        if dataset_name:
            overrides.setdefault("basic_unit", {}).setdefault("dataset", {}).setdefault("name", dataset_name)

        return overrides

    def _write_results(self, out_file: Path, results: Dict[str, Any]) -> None:
        wrote_results = False
        for plan in self.writer_plans:
            if plan.write_results:
                with out_file.open("w", encoding="utf-8") as handle:
                    json.dump(results, handle, indent=2)
                logger.info("💾 [%s] Saved results → %s", plan.name, out_file)
                wrote_results = True
        if not wrote_results:
            with out_file.open("w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
            logger.info("💾 Saved results → %s", out_file)


class InteractionModelStage:
    """Stage for training BERT-INT interaction model (phase 2)."""

    def __init__(self, resume: bool = False):
        """Initialize interaction model stage.

        Args:
            resume: Whether to resume from cached results
        """
        self.resume = resume

    def execute(
        self,
        dataset: Dataset,
        basic_unit_checkpoint_dir: Path,
        stage_cfg: Dict[str, Any],
        lineage_cfg: Dict[str, Any],
        ratio_root: Path,
        augmentation_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute interaction model training and evaluation.

        Args:
            dataset: Dataset to use (must have attribute data)
            basic_unit_checkpoint_dir: Directory containing basic_unit checkpoint
            stage_cfg: Stage configuration
            lineage_cfg: Lineage tracking configuration
            ratio_root: Root directory for this reduction ratio
            augmentation_name: Optional augmentation variant name

        Returns:
            Dictionary containing results and paths
        """
        import pickle
        import numpy as np
        import torch
        from transformers import BertTokenizer

        from src.alignment_models.methods.bert_int.basic_unit.model import BasicBertUnit
        from src.alignment_models.methods.bert_int.interaction_model import (
            AttributeValueCleaner,
            CandidateGenerator,
            InteractionDataset,
            NeighborViewFeatureExtractor,
            AttributeViewFeatureExtractor,
            DescriptionViewFeatureExtractor,
            InteractionMLP,
            InteractionTrainer,
        )

        # Get interaction model config
        # BERT-INT is always a two-phase model - defaults are loaded from config/models/bert_int.yaml
        # and merged with overrides from experiment config
        from src.alignment_models.methods.bert_int.config import load_bert_int_config

        # Load full config with defaults + overrides
        full_config = load_bert_int_config(overrides=stage_cfg)
        interaction_cfg = full_config.get("interaction_model", {})

        logger.info("=" * 80)
        logger.info("Starting BERT-INT Interaction Model (Phase 2)")
        logger.info("=" * 80)

        # Setup directories
        variant_key = augmentation_name if augmentation_name else "baseline"
        interaction_root = ratio_root / "interaction_model" / variant_key
        _ensure_directory(interaction_root)

        checkpoint_path = interaction_root / "interaction_model.pt"
        results_path = interaction_root / "results.json"

        # Check if we can resume
        if self.resume and results_path.exists():
            logger.info("⏭️  Skipping interaction model — results already cached")
            with results_path.open("r") as f:
                return json.load(f)

        # Configuration parameters (defaults loaded from config/models/bert_int.yaml)
        kernel_num = interaction_cfg["kernel_num"]
        max_neighbors = interaction_cfg["entity_neigh_max_num"]
        max_values = interaction_cfg["entity_attvalue_max_num"]
        candidate_topk = interaction_cfg["candidate_topk"]
        mlp_hidden_dim = interaction_cfg["mlp_hidden_dim"]
        epochs = interaction_cfg["epochs"]
        neg_num = interaction_cfg["neg_num"]
        margin = interaction_cfg["margin"]
        learning_rate = interaction_cfg["learning_rate"]
        batch_size = interaction_cfg["batch_size"]
        eval_every = interaction_cfg["eval_every"]
        device_str = interaction_cfg["device"]
        seed = stage_cfg.get("experiment", {}).get("seed")

        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load basic_unit model and other data
        logger.info("Loading basic_unit model and data...")
        basic_unit_other_data_path = basic_unit_checkpoint_dir / f"{variant_key}_other_data.pkl"

        if not basic_unit_other_data_path.exists():
            raise FileNotFoundError(
                f"Basic unit other_data not found at {basic_unit_other_data_path}. "
                f"Make sure basic_unit training completed successfully."
            )

        with open(basic_unit_other_data_path, "rb") as f:
            train_ill, test_ill, eid2data = pickle.load(f)

        logger.info(f"Train ILL: {len(train_ill)} pairs")
        logger.info(f"Test ILL: {len(test_ill)} pairs")

        # Find and load basic_unit model checkpoint
        checkpoint_files = sorted(basic_unit_checkpoint_dir.glob(f"{variant_key}_model_epoch_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No basic_unit checkpoint found in {basic_unit_checkpoint_dir}"
            )

        # Load the last checkpoint
        basic_unit_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading basic_unit checkpoint: {basic_unit_checkpoint}")

        # Get basic_unit config (defaults loaded from config/models/bert_int.yaml)
        basic_unit_cfg = full_config["basic_unit"]
        output_dim = basic_unit_cfg.get("output_dim") or basic_unit_cfg.get("result_size", 300)

        # Create config for BasicBertUnit
        basic_unit_config = {
            "result_size": output_dim,
            "encoder_name": basic_unit_cfg.get("encoder_name", "bert-base-multilingual-cased"),
        }
        basic_unit_model = BasicBertUnit(basic_unit_config)
        basic_unit_model.load_state_dict(torch.load(basic_unit_checkpoint, map_location="cpu"))
        basic_unit_model.eval()
        for param in basic_unit_model.parameters():
            param.requires_grad = False
        basic_unit_model = basic_unit_model.to(device)

        # Generate entity embeddings
        logger.info("Generating entity embeddings from basic_unit model...")
        entity_embeddings = []
        for eid in range(len(eid2data)):
            token_input = torch.LongTensor([eid2data[eid][0]]).to(device)
            mask_input = torch.FloatTensor([eid2data[eid][1]]).to(device)
            with torch.no_grad():
                vec = basic_unit_model(token_input, mask_input)
            entity_embeddings.append(vec.cpu().numpy()[0])

        entity_embeddings = np.array(entity_embeddings)
        logger.info(f"Entity embeddings shape: {entity_embeddings.shape}")

        # Generate candidates
        logger.info("Generating candidate entity pairs...")
        candidate_gen = CandidateGenerator(topk=candidate_topk, device=device)

        train_ids_1 = [e1 for e1, e2 in train_ill]
        train_ids_2 = [e2 for e1, e2 in train_ill]
        test_ids_1 = [e1 for e1, e2 in test_ill]
        test_ids_2 = [e2 for e1, e2 in test_ill]

        train_candidates = candidate_gen.generate(train_ids_1, train_ids_2, entity_embeddings)
        test_candidates = candidate_gen.generate(test_ids_1, test_ids_2, entity_embeddings)

        # Generate all entity pairs
        entity_pairs = InteractionDataset.generate_all_entity_pairs(
            [train_candidates, test_candidates],
            [train_ill]
        )

        # Extract neighbor-view features
        logger.info("Extracting neighbor-view interaction features...")
        # Get relational triples from dataset
        rel_triples = []
        for subj, pred, obj in dataset.knowledge_graph_source:
            rel_triples.append((int(subj), int(pred), int(obj)))
        for subj, pred, obj in dataset.knowledge_graph_target:
            rel_triples.append((int(subj), int(pred), int(obj)))

        # Add PAD entity
        pad_entity_id = len(entity_embeddings)
        entity_embeddings_padded = np.vstack([
            entity_embeddings,
            np.zeros((1, entity_embeddings.shape[1]))
        ])

        neighbor_extractor = NeighborViewFeatureExtractor(
            kernel_num=kernel_num,
            max_neighbors=max_neighbors,
            device=device
        )
        neighbor_dict = neighbor_extractor.build_neighbor_dict(rel_triples, pad_entity_id)
        neighbor_features = neighbor_extractor.extract_features(
            entity_pairs,
            entity_embeddings_padded,
            neighbor_dict,
            pad_entity_id,
            batch_size=2048
        )

        # Extract description-view features
        logger.info("Extracting description-view interaction features...")
        description_extractor = DescriptionViewFeatureExtractor(device=device)
        description_features = description_extractor.extract_features(
            entity_pairs,
            entity_embeddings,
            batch_size=512
        )

        # Extract attribute-view features
        logger.info("Extracting attribute-view interaction features...")

        # Clean attribute triples first
        # Note: This assumes dataset has attribute data available
        # You may need to load this from the dataset's attribute files
        logger.info("Loading and cleaning attribute triples...")

        # This is a placeholder - you'll need to load actual attribute data from dataset
        # For now, we'll create minimal attribute features
        # TODO: Implement proper attribute data loading from dataset
        attribute_features = np.zeros((len(entity_pairs), kernel_num * 2))
        logger.warning("Attribute features not fully implemented - using placeholder zeros")

        # Concatenate all features
        logger.info("Concatenating all interaction features...")
        all_features = np.concatenate([
            neighbor_features,
            attribute_features,
            description_features
        ], axis=1)

        logger.info(f"Final feature shape: {all_features.shape}")

        # Create interaction dataset
        interaction_dataset = InteractionDataset(
            entity_pairs=entity_pairs,
            features=all_features,
            train_ill=train_ill,
            test_ill=test_ill,
            train_candidates=train_candidates,
            test_candidates=test_candidates
        )

        # Create and train model
        logger.info("Creating interaction MLP model...")
        feature_dim = all_features.shape[1]
        model = InteractionMLP(input_dim=feature_dim, hidden_dim=mlp_hidden_dim)

        trainer = InteractionTrainer(
            model=model,
            dataset=interaction_dataset,
            device=device,
            learning_rate=learning_rate,
            margin=margin,
            neg_num=neg_num,
            batch_size=batch_size,
            seed=seed
        )

        # Train
        logger.info(f"Training interaction model for {epochs} epochs...")
        training_results = trainer.train(
            epochs=epochs,
            eval_every=eval_every,
            save_path=checkpoint_path
        )

        # Final evaluation
        logger.info("Final evaluation...")
        final_results = trainer.evaluator.evaluate(topk=candidate_topk)

        results = {
            "training": training_results,
            "final_evaluation": final_results,
            "config": {
                "kernel_num": kernel_num,
                "max_neighbors": max_neighbors,
                "candidate_topk": candidate_topk,
                "mlp_hidden_dim": mlp_hidden_dim,
                "epochs": epochs,
                "neg_num": neg_num,
                "margin": margin,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            },
            "checkpoint_path": str(checkpoint_path),
        }

        # Save results
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 80)
        logger.info("Interaction Model Training Completed!")
        logger.info(f"Best Hits@1: {training_results['best_hits@1']:.2f}%")
        logger.info("=" * 80)

        return results
