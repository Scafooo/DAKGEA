"""Core experiment orchestration logic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.loader import Config
from src.core.dataset import Dataset
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory
from src.logger import get_logger

logger = get_logger(__name__)

from .progress import ProgressTracker
from .specs import DatasetSpec, WriterPlan
from .stages import (
    ATTRIBUTE_DATA_DIR,
    MODEL_BERT_INT,
    RATIO_THRESHOLD,
    VARIANT_BASELINE,
    VARIANT_REDUCED,
    AugmentationStage,
    EvaluationStage,
    ReductionStage,
)


class ExperimentRunner:
    """Coordinate reduction, augmentation, and model evaluation for an experiment suite."""

    def __init__(
        self,
        exp_cfg: Dict[str, Any],
        overwrite_existing: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.exp_cfg = exp_cfg
        self.show_progress = show_progress

        # Auto-disable progress bar if logging is not ERROR level
        root_logger = logging.getLogger("KG_EA")
        if root_logger.level < logging.ERROR and self.show_progress:
            logger.debug("Disabling progress bar due to verbose logging (level < ERROR)")
            self.show_progress = False

        global_cfg = Config().get()
        self.paths = global_cfg["paths"]
        default_overwrite = global_cfg.get("experiment_defaults", {}).get(
            "overwrite_existing", False
        )

        try:
            self.name = exp_cfg["name"]
        except KeyError as exc:
            raise KeyError(
                f"Missing required experiment configuration key: 'name'. "
                f"Available keys: {list(exp_cfg.keys())}"
            ) from exc

        # Normalise dataset configuration to a list of entries
        if "datasets" in exp_cfg:
            datasets_cfg = exp_cfg["datasets"]
            if not isinstance(datasets_cfg, (list, tuple)):
                datasets_cfg = [datasets_cfg]
        elif "dataset" in exp_cfg:
            datasets_cfg = [exp_cfg["dataset"]]
        else:
            raise KeyError(
                f"Experiment configuration must define 'dataset' or 'datasets'. "
                f"Available keys: {list(exp_cfg.keys())}"
            )
        self.datasets_cfg = list(datasets_cfg)

        # Normalise reduction ratios to a list of floats
        if "reduction_ratios" in exp_cfg:
            ratios = exp_cfg["reduction_ratios"]
            if not isinstance(ratios, (list, tuple)):
                ratios = [ratios]
        elif "reduction_ratio" in exp_cfg:
            ratios = [exp_cfg["reduction_ratio"]]
        else:
            raise KeyError(
                f"Experiment configuration must define 'reduction_ratio' or 'reduction_ratios'. "
                f"Available keys: {list(exp_cfg.keys())}"
            )
        self.ratios = [float(r) for r in ratios]

        # Single augmentation method (optional)
        if "augmentation_methods" in exp_cfg:
            augmentations = exp_cfg.get("augmentation_methods", [])
            if not isinstance(augmentations, (list, tuple)):
                augmentations = [augmentations]
        elif "augmentation_method" in exp_cfg:
            augmentations = [exp_cfg["augmentation_method"]] if exp_cfg["augmentation_method"] else []
        else:
            augmentations = []
        self.augmentations = [a for a in augmentations if a]

        # Single evaluation model (required)
        if "models_to_run" in exp_cfg:
            models = exp_cfg["models_to_run"]
            if not isinstance(models, (list, tuple)):
                models = [models]
        elif "model" in exp_cfg:
            models = [exp_cfg["model"]]
        else:
            raise KeyError(
                f"Experiment configuration must define 'model' or 'models_to_run'. "
                f"Available keys: {list(exp_cfg.keys())}"
            )
        self.models = [m for m in models if m]

        self.reduction_method = exp_cfg.get("reduction_method", "random_entities")

        effective_overwrite = (
            overwrite_existing
            if overwrite_existing is not None
            else exp_cfg.get("overwrite_existing")
        )
        if effective_overwrite is None:
            effective_overwrite = default_overwrite
        self.overwrite_existing = bool(effective_overwrite)
        self.resume = not self.overwrite_existing

        self.base_data: Path = Path(self.paths["raw_data"])
        self.external_data: Path = Path(
            self.paths.get("external_data", self.base_data.parent / "external")
        )

        workspace_root = Path(self.paths.get("results", "results")) / self.name
        self.base_workspace: Path = workspace_root
        self.metadata_file: Path = self.base_workspace / "metadata.json"
        self.base_workspace.mkdir(parents=True, exist_ok=True)

        self.metadata: Dict[str, Any] = {
            "name": self.name,
            "reduction_method": self.reduction_method,
            "ratios": self.ratios,
            "augmentations": self.augmentations,
            "models": self.models,
            "overwrite_existing": self.overwrite_existing,
            "datasets": {},
            "workspace_root": str(self.base_workspace.resolve()),
        }

        self.datasets: List[DatasetSpec] = self._build_dataset_specs()

    def _infer_reader(self, dataset_name: str) -> str:
        """Guess the reader type based on the raw data directory structure."""
        available_dirs = []
        for candidate in self.base_data.iterdir():
            if not candidate.is_dir():
                continue
            available_dirs.append(candidate.name)
            if (candidate / dataset_name).is_dir():
                logger.debug(
                    "Auto-detected reader '%s' for dataset '%s'",
                    candidate.name,
                    dataset_name,
                )
                return candidate.name
        raise FileNotFoundError(
            f"Unable to infer reader for dataset '{dataset_name}'. "
            f"Expected to find it under {self.base_data}. "
            f"Available reader directories: {available_dirs}"
        )

    def run(self) -> None:
        """Execute the experiment suite over the configured datasets and ratios."""
        progress = ProgressTracker(
            total=len(self.datasets) * len(self.ratios), enabled=self.show_progress
        )
        logger.info(
            "=== Starting experiment suite '%s' (resume=%s, overwrite=%s) ===",
            self.name,
            self.resume,
            self.overwrite_existing,
        )

        try:
            for spec in self.datasets:
                dataset_root = self.base_data / spec.reader / spec.name
                dataset_workspace, dataset_meta = self._prepare_dataset_workspace(
                    spec, dataset_root
                )

                reader = DatasetReaderFactory.create_reader(spec.reader)
                dataset = reader.read(str(dataset_root), subtype=spec.subtype)
                writer_plans = self._resolve_writer_plans(spec)
                reduction_stage = ReductionStage(
                    self.reduction_method,
                    writer_plans,
                    self.resume,
                )
                augmentation_stage = AugmentationStage(
                    writer_plans,
                    self.resume,
                )
                evaluation_stage = EvaluationStage(
                    writer_plans,
                    self.models,
                    self.resume,
                )

                logger.info(
                    "→ Dataset '%s' (reader=%s, subtype=%s)",
                    spec.name,
                    spec.reader,
                    spec.subtype,
                )
                logger.info("[STEP] Preparing dataset '%s'", spec.name)

                single_ratio = len(self.ratios) == 1
                for ratio in self.ratios:
                    ratio_desc = f"{ratio * 100:.1f}%"
                    ratio_tag = self._format_ratio_tag(ratio)
                    if single_ratio:
                        ratio_root = dataset_workspace
                    else:
                        ratio_root = dataset_workspace / ratio_tag
                        ratio_root.mkdir(parents=True, exist_ok=True)

                    progress.set_description(f"📦 {spec.name} [{ratio_desc}]")
                    logger.info("[STEP] Ratio %.1f%% for dataset '%s'", ratio * 100, spec.name)
                    stage_cfg = self._build_stage_config(
                        spec.name,
                        len(dataset.aligned_entities),
                        ratio,
                        ratio_tag,
                        dataset_workspace,
                        ratio_root,
                        dataset_root,
                    )
                    ratio_meta = dataset_meta["ratios"].setdefault(ratio_tag, {})
                    ratio_meta.update(
                        {
                            "ratio": ratio,
                            "target_entities": stage_cfg["reduction"]["target_entities"],
                        }
                    )
                    reduction_meta = ratio_meta.setdefault(
                        "reduction",
                        {"method": self.reduction_method, "paths": {}},
                    )
                    reduction_meta["method"] = self.reduction_method
                    reduction_meta.setdefault("paths", {})
                    ratio_meta.setdefault("augmentations", {})
                    ratio_meta.setdefault("evaluations", {})

                    lineage = stage_cfg.setdefault("lineage", {})
                    lineage["_reduction_executed"] = False

                    # Execute reduction or use raw dataset
                    dataset_reduced = self._execute_reduction_if_needed(
                        ratio, dataset, reader, reduction_stage, stage_cfg,
                        ratio_tag, lineage, ratio_root, ratio_meta, dataset_root, spec
                    )

                    # Execute baseline and augmented evaluations
                    self._execute_evaluations(
                        dataset_reduced, augmentation_stage, evaluation_stage,
                        stage_cfg, lineage, ratio, ratio_tag, ratio_root, ratio_meta,
                        reader, spec
                    )

                    progress.step()
        finally:
            progress.close()
            self._write_metadata()
            logger.info("=== All experiments completed ===")

    def _build_dataset_specs(self) -> List[DatasetSpec]:
        """Expand dataset entries into normalized specifications."""
        default_writer = self.exp_cfg.get("writers", self.exp_cfg.get("writer"))
        default_readers: Dict[str, str] = self.exp_cfg.get("readers", {})
        default_subtype = self.exp_cfg.get("dataset_type", "attribute_data")

        specs: List[DatasetSpec] = []
        for entry in self.datasets_cfg:
            if isinstance(entry, dict):
                name = entry["name"]
                reader = entry.get("reader") or entry.get("reader_type") or default_readers.get(name)
                subtype = entry.get("subtype", default_subtype)
                writer_conf = entry.get("writers", entry.get("writer", default_writer))
            else:
                name = str(entry)
                reader = default_readers.get(name)
                subtype = default_subtype
                writer_conf = default_writer

            if reader is None:
                reader = self._infer_reader(name)

            specs.append(
                DatasetSpec(
                    name=name,
                    reader=reader,
                    subtype=subtype,
                    writer_conf=writer_conf,
                )
            )
        return specs

    def _prepare_dataset_workspace(
        self,
        spec: DatasetSpec,
        dataset_root: Path,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Create or refresh the per-dataset workspace within the experiment folder."""
        dataset_workspace = self.base_workspace / spec.name
        dataset_workspace.mkdir(parents=True, exist_ok=True)

        dataset_meta = self.metadata["datasets"].setdefault(
            spec.name,
            {
                "reader": spec.reader,
                "subtype": spec.subtype,
                "raw_source": str(dataset_root.resolve()),
                "workspace": str(dataset_workspace.resolve()),
                "ratios": {},
            },
        )
        dataset_meta["reader"] = spec.reader
        dataset_meta["subtype"] = spec.subtype
        dataset_meta["raw_source"] = str(dataset_root.resolve())
        dataset_meta["workspace"] = str(dataset_workspace.resolve())

        return dataset_workspace, dataset_meta

    def _resolve_writer_plans(self, spec: DatasetSpec) -> List[WriterPlan]:
        """Instantiate writers based on dataset or global writer configuration."""
        conf = spec.writer_conf
        if conf is None:
            return []

        if not isinstance(conf, (list, tuple)):
            conf = [conf]

        plans: List[WriterPlan] = []
        for entry in conf:
            if isinstance(entry, str):
                writer_type = entry
                settings = {}
            elif isinstance(entry, dict):
                writer_type = entry.get("type", spec.reader)
                settings = entry
            else:
                raise ValueError(f"Unsupported writer configuration: {entry!r}")

            writer = DatasetWriterFactory.create_writer(writer_type)
            plans.append(
                WriterPlan(
                    name=writer_type,
                    writer=writer,
                    write_reduced=settings.get("write_reduced", True),
                    write_augmented=settings.get("write_augmented", True),
                    write_results=settings.get("write_results", True),
                )
            )
        return plans

    def _build_stage_config(
        self,
        dataset_name: str,
        total_entities: int,
        ratio: float,
        ratio_tag: str,
        dataset_workspace: Path,
        ratio_root: Path,
        dataset_root: Path,
    ) -> Dict[str, Any]:
        """Create a configuration payload passed to reducers, augmenters, and models."""
        cfg = dict(self.exp_cfg.get("parameters", {}))
        reduction_cfg = cfg.setdefault("reduction", {})
        reduction_cfg["target_entities"] = max(1, int(total_entities * ratio))
        experiment_meta = cfg.setdefault("experiment", {})
        experiment_meta.update(
            {
                "name": self.name,
                "dataset": dataset_name,
                "ratio": ratio,
                "ratio_tag": ratio_tag,
                "reduction_method": self.reduction_method,
            }
        )

        # Copy model-specific configurations from experiment config
        # Note: self.exp_cfg is already the content under "experiment" key from YAML
        # Copy basic_unit config if present
        if "basic_unit" in self.exp_cfg:
            cfg["basic_unit"] = self.exp_cfg["basic_unit"]
        # Copy interaction_model config if present
        if "interaction_model" in self.exp_cfg:
            cfg["interaction_model"] = self.exp_cfg["interaction_model"]
        # Copy model config if present (only if it's a dict with sub-config)
        if "model" in self.exp_cfg and isinstance(self.exp_cfg["model"], dict):
            cfg.setdefault("model", {}).update(self.exp_cfg["model"])
        # Copy seed if present
        if "seed" in self.exp_cfg:
            cfg["seed"] = self.exp_cfg["seed"]

        lineage = cfg.setdefault("lineage", {})
        lineage["reduction_method"] = self.reduction_method
        lineage["dataset_workspace"] = str(dataset_workspace.resolve())
        lineage["ratio_tag"] = ratio_tag
        lineage["ratio_root"] = str(ratio_root.resolve())
        lineage["raw_source"] = str(dataset_root.resolve())
        reduction_root = ratio_root / "reduction"
        augmentation_root = ratio_root / "augmentation"
        evaluation_root = ratio_root / "evaluation"
        lineage["reduction_root"] = str(reduction_root.resolve())
        lineage["reduced_base"] = str(reduction_root.resolve())
        lineage["augmentation_root"] = str(augmentation_root.resolve())
        lineage["augmented_base"] = str(augmentation_root.resolve())
        lineage["evaluation_root"] = str(evaluation_root.resolve())
        lineage.setdefault("evaluation_dirs", {})
        lineage["reduced_paths"] = lineage.get("reduced_paths", {})
        lineage["augmented_paths"] = lineage.get("augmented_paths", {})
        lineage["augmented_hybea_paths"] = lineage.get("augmented_hybea_paths", {})
        logger.debug(
            "Stage config prepared for dataset=%s ratio=%.3f (target_entities=%d)",
            dataset_name,
            ratio,
            reduction_cfg["target_entities"],
        )
        logger.debug("[IMPORTANT] Lineage tracking: %s", lineage)
        return cfg

    @staticmethod
    def _format_ratio_tag(ratio: float) -> str:
        """Return a normalized percentage tag (e.g., 0.1 -> '10')."""
        return str(int(round(ratio * 100)))

    def _execute_reduction_if_needed(
        self,
        ratio: float,
        dataset,
        reader,
        reduction_stage,
        stage_cfg: Dict[str, Any],
        ratio_tag: str,
        lineage: Dict[str, Any],
        ratio_root: Path,
        ratio_meta: Dict[str, Any],
        dataset_root: Path,
        spec,
    ):
        """
        Execute reduction stage if ratio < threshold, otherwise use raw dataset.

        When ratio is less than RATIO_THRESHOLD (0.999999), this method executes
        the reduction stage to create a reduced dataset. Otherwise, it clones
        the original dataset and uses it as-is.

        Args:
            ratio: Reduction ratio (0.0 to 1.0)
            dataset: Original dataset to reduce
            reader: Dataset reader instance
            reduction_stage: ReductionStage instance
            stage_cfg: Stage configuration dictionary
            ratio_tag: String tag for the ratio (e.g., "10" for 0.1)
            lineage: Lineage tracking dictionary
            ratio_root: Root directory for this ratio's outputs
            ratio_meta: Metadata dictionary for this ratio
            dataset_root: Root directory of the original dataset
            spec: DatasetSpec instance

        Returns:
            Dataset: Reduced or cloned dataset
        """
        reduced_datasets = lineage.setdefault("reduced_datasets", {})

        if ratio < RATIO_THRESHOLD:  # perform reduction only when ratio < 1
            dataset_reduced = reduction_stage.execute(
                stage_cfg,
                dataset,
                reader,
                ratio,
                ratio_tag,
                lineage,
                ratio_root,
                ratio_meta,
                spec.subtype,
            )
            lineage["_reduction_executed"] = True
        else:
            dataset_reduced = dataset.clone()
            raw_attr = dataset_root / ATTRIBUTE_DATA_DIR
            dataset_path = raw_attr if raw_attr.exists() else dataset_root
            reduced_datasets["raw"] = str(dataset_path.resolve())
            lineage.setdefault("reduced_paths", {})["raw"] = str(dataset_path.resolve())
            reduction_meta = ratio_meta.setdefault(
                "reduction",
                {"method": self.reduction_method, "paths": {}},
            )
            reduction_meta["method"] = self.reduction_method
            reduction_meta.setdefault("paths", {})

        return dataset_reduced

    def _execute_evaluations(
        self,
        dataset_reduced,
        augmentation_stage,
        evaluation_stage,
        stage_cfg: Dict[str, Any],
        lineage: Dict[str, Any],
        ratio: float,
        ratio_tag: str,
        ratio_root: Path,
        ratio_meta: Dict[str, Any],
        reader,
        spec,
    ) -> None:
        """
        Execute baseline and augmentation evaluations.

        If no augmentation methods are configured, evaluates the baseline (reduced)
        dataset only. Otherwise, iterates through all configured augmentation methods,
        augments the dataset, and evaluates each augmented variant.

        For BERT-INT model with interaction_model enabled, also runs the interaction
        model stage after basic_unit evaluation.

        Args:
            dataset_reduced: The reduced (or full) dataset to evaluate
            augmentation_stage: AugmentationStage instance
            evaluation_stage: EvaluationStage instance
            stage_cfg: Stage configuration dictionary
            lineage: Lineage tracking dictionary
            ratio: Reduction ratio
            ratio_tag: String tag for the ratio
            ratio_root: Root directory for this ratio's outputs
            ratio_meta: Metadata dictionary for this ratio
            reader: Dataset reader instance
            spec: DatasetSpec instance
        """
        from .stages import InteractionModelStage

        # Check if we need to run interaction model
        should_run_interaction = self._should_run_interaction_model(stage_cfg)

        if not self.augmentations:
            evaluation_stage.execute(
                VARIANT_BASELINE,
                dataset_reduced,
                dataset_reduced,
                stage_cfg,
                lineage,
                ratio_root,
                ratio_tag,
                ratio_meta,
            )

            # Run interaction model for baseline if applicable
            if should_run_interaction:
                self._run_interaction_model_stage(
                    dataset_reduced,
                    stage_cfg,
                    lineage,
                    ratio_root,
                    augmentation_name=None,
                )

        for aug_name in self.augmentations:
            dataset_augmented = augmentation_stage.execute(
                stage_cfg,
                aug_name,
                dataset_reduced,
                reader,
                lineage,
                ratio,
                ratio_tag,
                ratio_root,
                ratio_meta,
                spec.subtype,
            )
            evaluation_stage.execute(
                aug_name,
                dataset_reduced,
                dataset_augmented,
                stage_cfg,
                lineage,
                ratio_root,
                ratio_tag,
                ratio_meta,
            )

            # Run interaction model for this augmentation if applicable
            if should_run_interaction:
                self._run_interaction_model_stage(
                    dataset_augmented,
                    stage_cfg,
                    lineage,
                    ratio_root,
                    augmentation_name=aug_name,
                )

    def _should_run_interaction_model(self, stage_cfg: Dict[str, Any]) -> bool:
        """Check if interaction model should be run based on model type.

        For BERT-INT, interaction_model is always run as it's an integral
        part of the two-phase architecture (basic_unit → interaction_model).

        Args:
            stage_cfg: Stage configuration

        Returns:
            True if model is bert_int, False otherwise
        """
        # BERT-INT is always a two-phase model
        return MODEL_BERT_INT in self.models

    def _run_interaction_model_stage(
        self,
        dataset: Dataset,
        stage_cfg: Dict[str, Any],
        lineage: Dict[str, Any],
        ratio_root: Path,
        augmentation_name: Optional[str] = None,
    ) -> None:
        """Run the interaction model stage for BERT-INT.

        Args:
            dataset: Dataset to use
            stage_cfg: Stage configuration
            lineage: Lineage tracking
            ratio_root: Root directory for this ratio
            augmentation_name: Optional augmentation name (None for baseline)
        """
        from .stages import InteractionModelStage

        logger.info("[STEP] → Running BERT-INT Interaction Model (Phase 2)")

        # Determine checkpoint directory based on variant
        variant_key = augmentation_name if augmentation_name else "baseline"
        evaluation_dirs = lineage.get("evaluation_dirs", {})

        # For baseline, use "reduced" as key
        if variant_key == "baseline":
            variant_key_for_eval = VARIANT_REDUCED
        else:
            variant_key_for_eval = variant_key.replace("/", "_")

        evaluation_dir = evaluation_dirs.get(variant_key_for_eval)

        if not evaluation_dir:
            logger.warning(
                f"Evaluation directory not found for variant '{variant_key_for_eval}', "
                f"skipping interaction model"
            )
            return

        # The checkpoint directory should be under the evaluation directory
        checkpoint_dir = Path(evaluation_dir)

        # Create and execute interaction model stage
        interaction_stage = InteractionModelStage(resume=self.resume)

        try:
            results = interaction_stage.execute(
                dataset=dataset,
                basic_unit_checkpoint_dir=checkpoint_dir,
                stage_cfg=stage_cfg,
                lineage_cfg=lineage,
                ratio_root=ratio_root,
                augmentation_name=augmentation_name,
            )

            logger.info("[SUCCESS] Interaction model completed successfully")

            # Log key metrics if available
            if results and "final_evaluation" in results:
                eval_results = results["final_evaluation"]
                logger.info(
                    f"  Hits@1: {eval_results.get('hits@1', 0):.2f}%  "
                    f"Hits@10: {eval_results.get('hits@10', 0):.2f}%  "
                    f"MRR: {eval_results.get('mrr', 0):.4f}"
                )

                # Update bert_int.json with interaction model results (final results)
                self._update_bert_int_results_with_interaction(
                    checkpoint_dir, eval_results
                )

        except Exception as e:
            logger.error(f"Interaction model failed: {e}", exc_info=True)
            logger.warning("Continuing without interaction model results")

    def _update_bert_int_results_with_interaction(
        self,
        evaluation_dir: Path,
        interaction_results: Dict[str, Any],
    ) -> None:
        """Update bert_int.json with final interaction model results.

        This makes the interaction model results the "official" BERT-INT results,
        while preserving the basic_unit results for reference.

        Args:
            evaluation_dir: Evaluation directory containing bert_int.json
            interaction_results: Results from interaction model evaluation
        """
        bert_int_json = evaluation_dir / "bert_int.json"

        if not bert_int_json.exists():
            logger.warning(f"bert_int.json not found at {bert_int_json}, skipping update")
            return

        # Load existing results (basic_unit)
        with bert_int_json.open("r") as f:
            basic_unit_results = json.load(f)

        # Create comprehensive results combining both phases
        final_results = {
            "model": "bert_int",
            "phases": {
                "basic_unit": basic_unit_results,  # Preserve original basic_unit results
                "interaction_model": interaction_results,  # Add interaction model results
            },
            # Top-level metrics are from interaction model (final results)
            "hits@1": interaction_results.get("hits@1", 0.0),
            "hits@5": interaction_results.get("hits@5", 0.0),
            "hits@10": interaction_results.get("hits@10", 0.0),
            "hits@25": interaction_results.get("hits@25", 0.0),
            "hits@50": interaction_results.get("hits@50", 0.0),
            "mr": interaction_results.get("mr", 0.0),
            "mrr": interaction_results.get("mrr", 0.0),
            "evaluated": interaction_results.get("total", 0),
            # Add note about the two-phase nature
            "_note": "BERT-INT is a two-phase model: basic_unit (phase 1) + interaction_model (phase 2). "
                     "Top-level metrics are from interaction_model (final results).",
        }

        # Write updated results
        with bert_int_json.open("w") as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"✓ Updated {bert_int_json.name} with interaction model results (final)")
        logger.info(
            f"  Basic unit results preserved in phases.basic_unit"
        )

    def _write_metadata(self) -> None:
        """Persist the experiment metadata summary alongside artefacts."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)
        logger.info("🗒️  Experiment metadata saved → %s", self.metadata_file)
