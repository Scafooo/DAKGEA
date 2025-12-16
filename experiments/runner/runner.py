"""Core experiment orchestration logic."""

from __future__ import annotations

import copy
import json
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.loader import Config
from src.core.dataset import Dataset
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory
from src.logger import get_logger

# Suppress RDFLib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rdflib")
warnings.filterwarnings("ignore", message=".*rdflib.*")

# Suppress specific RDFLib loggers
logging.getLogger("rdflib").setLevel(logging.ERROR)
logging.getLogger("rdflib.term").setLevel(logging.ERROR)
logging.getLogger("rdflib.plugins").setLevel(logging.ERROR)

logger = get_logger(__name__)

from .configuration import ExperimentConfig
from .progress import ProgressTracker
from .specs import DatasetSpec, WriterPlan
from .stages import (
    RATIO_THRESHOLD,
    VARIANT_BASELINE,
    VARIANT_REDUCED,
    AugmentationStage,
    EvaluationStage,
    FilteringStage,
    ReductionStage,
    StageSummaryWriter,
)


class ExperimentRunner:
    """Coordinate reduction, augmentation, and model evaluation for an experiment suite."""

    def __init__(
        self,
        exp_cfg: Dict[str, Any],
        overwrite_existing: Optional[bool] = None,
        show_progress: bool = True,
    ) -> None:
        self.exp_cfg = exp_cfg
        self.show_progress = show_progress

        # Auto-disable progress bar if logging is not ERROR level
        root_logger = logging.getLogger("KG_EA")
        if root_logger.level < logging.ERROR and self.show_progress:
            logger.debug("Disabling progress bar due to verbose logging (level < ERROR)")
            self.show_progress = False

        self.global_cfg = Config().get()
        self.paths = self.global_cfg["paths"]
        default_overwrite = self.global_cfg.get("experiment_defaults", {}).get(
            "overwrite_existing", False
        )

        self.normalized_cfg = ExperimentConfig.from_payload(
            exp_cfg,
            cli_overwrite=overwrite_existing,
            default_overwrite=default_overwrite,
        )
        self.name = self.normalized_cfg.name
        self.dataset_cfg = self.normalized_cfg.dataset
        self.datasets_cfg = [self.dataset_cfg]
        self.ratio_value = self.normalized_cfg.ratio
        self.ratios = [self.ratio_value] if self.ratio_value is not None else []
        augmentation_method = self.normalized_cfg.augmentation
        self.augmentations = [augmentation_method] if augmentation_method else []
        self.models = list(self.normalized_cfg.models)
        self.reduction_method = self.normalized_cfg.reduction_method
        self.reduction_writer = self.normalized_cfg.reduction_writer
        self.reduction_save_dataset = self.normalized_cfg.reduction_save_dataset
        self.reduction_save_model = self.normalized_cfg.reduction_save_model
        self.reduction_eval = self.normalized_cfg.reduction_eval
        self.augmentation_writer = self.normalized_cfg.augmentation_writer
        self.augmentation_save_dataset = self.normalized_cfg.augmentation_save_dataset
        self.augmentation_save_model = self.normalized_cfg.augmentation_save_model
        self.augmentation_eval = self.normalized_cfg.augmentation_eval
        self.clear_intermediate = self.normalized_cfg.clear_intermediate
        self.overwrite_existing = self.normalized_cfg.overwrite_existing
        self.resume = self.normalized_cfg.resume

        self.base_data: Path = Path(self.paths["raw_data"])
        self.external_data: Path = Path(
            self.paths.get("external_data", self.base_data.parent / "external")
        )

        # Build workspace path: results/[suite/]name
        workspace_root = Path(self.paths.get("results", "results"))
        if self.normalized_cfg.suite:
            workspace_root = workspace_root / self.normalized_cfg.suite
        workspace_root = workspace_root / self.name
        self.base_workspace: Path = workspace_root
        self.metadata_file: Path = self.base_workspace / "metadata.json"

        # EARLY EXIT: Skip initialization if experiment already complete and overwrite not requested
        if not self.overwrite_existing and self._check_early_completion_before_init():
            logger.info("✅ Experiment already completed, skipping (set overwrite_existing=true to re-run)")
            # Set minimal state for early exit
            self.metadata = {}
            self.datasets = []
            self._early_exit = True
            return

        self._early_exit = False
        self.base_workspace.mkdir(parents=True, exist_ok=True)

        self.metadata: Dict[str, Any] = {
            "name": self.name,
            "suite": self.normalized_cfg.suite,
            "reduction_method": self.reduction_method,
            "ratios": self.ratios,
            "augmentations": self.augmentations,
            "models": self.models,
            "overwrite_existing": self.overwrite_existing,
            "datasets": {},
            "workspace_root": str(self.base_workspace.resolve()),
        }

        self.datasets: List[DatasetSpec] = self._build_dataset_specs()

    def _infer_reader(self, dataset_name: str) -> Tuple[str, str]:
        """Guess the reader type based on the raw data directory structure.

        Supports two formats:
        1. Simple name: 'BBC_DB' -> searches for it under all reader directories
        2. Reader/dataset format: 'hybea/BBC_DB' or 'rdf/DW_15' -> uses explicit reader

        Returns:
            Tuple of (reader_name, actual_dataset_name)
        """
        # Check if dataset_name contains '/' (explicit reader/dataset format)
        if "/" in dataset_name:
            parts = dataset_name.split("/", 1)
            reader_name = parts[0]
            actual_dataset_name = parts[1]

            # Verify that the path exists
            dataset_path = self.base_data / reader_name / actual_dataset_name
            if dataset_path.is_dir():
                logger.debug(
                    "Using explicit reader '%s' for dataset '%s' (from '%s')",
                    reader_name,
                    actual_dataset_name,
                    dataset_name,
                )
                return reader_name, actual_dataset_name
            else:
                raise FileNotFoundError(
                    f"Dataset path not found: {dataset_path}. "
                    f"Expected format: data/raw/{reader_name}/{actual_dataset_name}"
                )

        # Original behavior: search for dataset under all reader directories
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
                return candidate.name, dataset_name
        raise FileNotFoundError(
            f"Unable to infer reader for dataset '{dataset_name}'. "
            f"Expected to find it under {self.base_data}. "
            f"Available reader directories: {available_dirs}"
        )

    def _infer_reader_from_direct_path(self, path: str) -> str:
        """Infer reader type from a direct dataset path by checking for known format indicators."""
        from pathlib import Path

        path_obj = Path(path)
        path_str = str(path_obj).lower()

        # Check for known reader types in path components
        known_readers = ["hybea", "rdf", "bert_int"]
        for reader in known_readers:
            if reader in path_str or reader.replace("_", "") in path_str:
                logger.debug(
                    "Auto-detected reader '%s' from direct path '%s'",
                    reader,
                    path,
                )
                return reader

        # Check for format-specific files to infer reader
        if path_obj.is_dir():
            files = [f.name for f in path_obj.iterdir()]

            # BERT-INT format: has ent_ids_1, ent_ids_2, sup_pairs, ref_pairs
            if "ent_ids_1" in files and "sup_pairs" in files:
                logger.debug("Detected BERT-INT format from file structure in '%s'", path)
                return "bert_int"

            # HybEA format: has rel_triples_1, rel_triples_2 or attr_triples1
            if "rel_triples_1" in files or "attr_triples1" in files or "triples_1" in files:
                logger.debug("Detected HybEA format from file structure in '%s'", path)
                return "hybea"

            # RDF format: has .ttl or .nt files
            for file in files:
                if file.endswith(".ttl") or file.endswith(".nt"):
                    logger.debug("Detected RDF format from file extensions in '%s'", path)
                    return "rdf"

        raise ValueError(
            f"Unable to infer reader type from path '{path}'. "
            f"Path should contain 'hybea', 'rdf', or 'bert_int', "
            f"or have recognizable file structure."
        )

    def run(self) -> None:
        """Execute the experiment suite over the configured datasets and ratios."""
        # EARLY EXIT: Already checked in __init__, just return if flagged
        if hasattr(self, '_early_exit') and self._early_exit:
            return

        # EARLY EXIT: Additional runtime check (in case of programmatic usage)
        if self._check_early_completion():
            logger.info("✅ Experiment already completed, skipping execution (set overwrite_existing=true to re-run)")
            return

        # Check if we're in direct path mode (no ratios = direct dataset access)
        direct_mode = self.normalized_cfg.direct_mode

        try:
            if direct_mode:
                self._run_direct_mode()
            else:
                self._run_standard_mode()
        finally:
            # Clean up intermediate files if requested (always execute, even on error)
            if self.clear_intermediate:
                self._cleanup_intermediate_files()

    def _run_direct_mode(self) -> None:
        """Run experiments in direct path mode: read datasets directly, skip reduction/writer."""
        logger.info(
            "=== Running in DIRECT PATH mode: '%s' ===",
            self.name,
        )
        logger.info("Skipping reduction, augmentation, and writer stages")

        progress = ProgressTracker(total=1, enabled=self.show_progress)

        pending_cleanup: List[Path] = []
        try:
            spec = self.datasets[0]
            direct_path = spec.direct_path
            if not direct_path:
                raise ValueError(
                    f"Direct path mode requires 'path' field in dataset config for '{spec.name}'"
                )

            dataset_root = Path(direct_path)
            logger.info(
                "→ Dataset '%s' (reader=%s, path=%s)",
                spec.name,
                spec.reader,
                dataset_root,
            )

            dataset_workspace, dataset_meta = self._prepare_dataset_workspace(
                spec, dataset_root
            )
            artifact_root = dataset_workspace / "artifact"
            artifact_root.mkdir(parents=True, exist_ok=True)
            dataset_meta["artifact_root"] = str(artifact_root.resolve())

            reader = DatasetReaderFactory.create_reader(spec.reader)
            dataset = reader.read(str(dataset_root))

            ratio_tag = "direct"
            ratio = 1.0
            ratio_root = dataset_workspace

            # Skip entire experiment if already fully completed
            if self._is_experiment_complete(dataset_workspace, artifact_root, ratio_tag):
                progress.step()
                return

            stage_cfg = self._build_stage_config(
                spec.name,
                len(dataset.aligned_entities),
                ratio,
                ratio_tag,
                dataset_workspace,
                ratio_root,
                dataset_root,
            )
            lineage = stage_cfg.setdefault("lineage", {})
            lineage.update(
                {
                    "direct_mode": True,
                    "dataset_workspace": str(dataset_root.resolve()),
                    "raw_source": str(dataset_root.resolve()),
                    "artifact_root": str(artifact_root.resolve()),
                    "reduction_root": str((dataset_workspace / "reduction").resolve()),
                    "evaluation_root": str((artifact_root / "evaluation").resolve()),
                }
            )

            ratio_meta = dataset_meta["ratios"].setdefault(ratio_tag, {})
            ratio_meta.update(
                {
                    "ratio": ratio,
                    "target_entities": len(dataset.aligned_entities),
                }
            )
            reduction_meta = ratio_meta.setdefault(
                "reduction",
                {"method": "direct", "paths": {}},
            )
            reduction_meta["source"] = str(dataset_root.resolve())
            ratio_meta.setdefault("augmentations", {})
            ratio_meta.setdefault("evaluations", {})

            reduction_root = dataset_workspace / "reduction"
            reduction_root.mkdir(parents=True, exist_ok=True)
            StageSummaryWriter.write(
                reduction_root / "summary.json",
                {
                    "method": "direct",
                    "ratio": ratio,
                    "target_entities": len(dataset.aligned_entities),
                    "aligned_pairs": len(dataset.aligned_entities),
                    "writer": None,
                },
            )

            evaluation_stage = EvaluationStage(
                self._resolve_writer_plans(spec),
                self.models,
                self.resume,
            )
            logger.info("[STEP] Running evaluation for '%s' (direct path)", spec.name)

            if self.reduction_eval:
                evaluation_stage.execute(
                    VARIANT_BASELINE,
                    dataset,
                    dataset,
                    stage_cfg,
                    lineage,
                    ratio_root,
                    ratio_tag,
                    ratio_meta,
                )
                persisted = self._persist_stage_results(
                    ratio_meta,
                    VARIANT_BASELINE,
                    reduction_root / "results.json",
                )
                if persisted:
                    reduction_meta["results"] = str(persisted)
            else:
                logger.info("⏭️  Skipping evaluation (reduction.eval=false)")

            progress.step()
            pending_cleanup = self._schedule_stage_cleanup()
        finally:
            progress.close()
            self._write_metadata()
            self._remove_stage_outputs(pending_cleanup)
            logger.info("=== Direct mode experiments completed ===")

    def _run_standard_mode(self) -> None:
        """Run experiments in standard mode with reduction, augmentation, and writer."""
        progress = ProgressTracker(total=1, enabled=self.show_progress)
        logger.info(
            "=== Starting experiment suite '%s' (resume=%s, overwrite=%s) ===",
            self.name,
            self.resume,
            self.overwrite_existing,
        )

        pending_cleanup: List[Path] = []
        try:
            spec = self.datasets[0]
            dataset_root = self.base_data / spec.reader / spec.name
            dataset_workspace, dataset_meta = self._prepare_dataset_workspace(
                spec, dataset_root
            )
            artifact_root = dataset_workspace / "artifact"
            artifact_root.mkdir(parents=True, exist_ok=True)
            dataset_meta["artifact_root"] = str(artifact_root.resolve())

            reader = DatasetReaderFactory.create_reader(spec.reader)
            dataset = reader.read(str(dataset_root))

            reduction_writer_plans = self._resolve_writer_plans(
                spec,
                override_writer=self.reduction_writer,
            )
            augmentation_writer_plans = self._resolve_writer_plans(
                spec,
                override_writer=self.augmentation_writer,
            )
            evaluation_writer_plans = self._resolve_writer_plans(spec)

            reduction_stage = ReductionStage(
                self.reduction_method,
                reduction_writer_plans,
                self.resume,
            )
            augmentation_stage = AugmentationStage(
                augmentation_writer_plans,
                self.resume,
            )
            evaluation_stage = EvaluationStage(
                evaluation_writer_plans,
                self.models,
                self.resume,
            )

            logger.info(
                "→ Dataset '%s' (reader=%s)",
                spec.name,
                spec.reader,
            )
            logger.info("[STEP] Preparing dataset '%s'", spec.name)

            ratio = self.ratio_value if self.ratio_value is not None else 1.0
            ratio_desc = f"{ratio * 100:.1f}%"
            ratio_tag = self._format_ratio_tag(ratio)
            ratio_root = dataset_workspace

            # Skip entire experiment if already fully completed
            if self._is_experiment_complete(dataset_workspace, artifact_root, ratio_tag):
                progress.step()
                return

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
            lineage = stage_cfg.setdefault("lineage", {})
            lineage.update(
                {
                    "artifact_root": str(artifact_root.resolve()),
                    "reduction_root": str((dataset_workspace / "reduction").resolve()),
                    "augmentation_root": str((dataset_workspace / "augmentation").resolve()),
                    "evaluation_root": str((artifact_root / "evaluation").resolve()),
                }
            )
            lineage["_reduction_executed"] = False

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

            # EARLY SKIP: Check if all results already exist and meet threshold BEFORE any execution
            retry_config = self.global_cfg.get("auto_retry_until_improvement", {})
            retry_enabled = retry_config.get("enabled", False)

            if retry_enabled and self.resume and not self.overwrite_existing:
                reduction_results = dataset_workspace / "reduction" / "results.json"
                augmentation_results = dataset_workspace / "augmentation" / "results.json"

                if reduction_results.exists() and augmentation_results.exists():
                    try:
                        metric = retry_config.get("metric", "hits@1")
                        min_improvement = retry_config.get("min_improvement", 0.01)

                        with reduction_results.open("r") as f:
                            red_res = json.load(f)
                        with augmentation_results.open("r") as f:
                            aug_res = json.load(f)

                        baseline_val = None
                        for m, mr in red_res.items():
                            if isinstance(mr, dict) and metric in mr:
                                baseline_val = mr[metric]
                                break

                        current_val = None
                        for m, mr in aug_res.items():
                            if isinstance(mr, dict) and metric in mr:
                                current_val = mr[metric]
                                break

                        if baseline_val is not None and current_val is not None:
                            improvement = current_val - baseline_val
                            if improvement >= min_improvement:
                                logger.info("⚡ SKIPPING ENTIRE RATIO (%.2f): Results exist and meet threshold (%.4f >= %.4f)",
                                          ratio, improvement, min_improvement)
                                # Register results in metadata
                                reduction_meta["results"] = str(reduction_results)
                                for aug_name in self.augmentations:
                                    augmentation_meta = ratio_meta["augmentations"].setdefault(aug_name, {})
                                    augmentation_meta["results"] = str(augmentation_results)
                                    augmentation_meta["skipped"] = True
                                    augmentation_meta["skipped_reason"] = "complete_skip_on_resume"
                                continue  # Skip this entire ratio
                    except (json.JSONDecodeError, OSError, KeyError):
                        pass  # Continue with normal execution

            dataset_reduced = self._execute_reduction_if_needed(
                ratio,
                dataset,
                reader,
                reduction_stage,
                stage_cfg,
                ratio_tag,
                lineage,
                ratio_root,
                ratio_meta,
                dataset_root,
                spec,
            )

            self._execute_evaluations(
                dataset,
                dataset_reduced,
                reduction_stage,
                augmentation_stage,
                evaluation_stage,
                stage_cfg,
                lineage,
                ratio,
                ratio_tag,
                ratio_root,
                ratio_meta,
                reader,
                spec,
                dataset_workspace,
            )
            progress.step()
            pending_cleanup = self._schedule_stage_cleanup()
        finally:
            progress.close()
            self._write_metadata()
            self._remove_stage_outputs(pending_cleanup)
            logger.info("=== All experiments completed ===")

    def _build_dataset_specs(self) -> List[DatasetSpec]:
        """Expand dataset entries into normalized specifications.

        Two modes supported:
        1. Standard: Reader inferred from data/raw/{reader}/{name}
        2. Direct: Read from specified path, skip reduction/writer

        Reader type is automatically inferred from the dataset path structure
        (e.g., data/raw/hybea/D_W_15K_V1 -> reader=hybea) unless explicitly
        specified in the configuration.
        """
        default_writer = self.exp_cfg.get("writers", self.exp_cfg.get("writer"))
        default_readers: Dict[str, str] = self.exp_cfg.get("readers", {})

        specs: List[DatasetSpec] = []
        for entry in self.datasets_cfg:
            direct_path = None

            if isinstance(entry, dict):
                # Check for direct path mode
                direct_path = entry.get("path")

                if direct_path:
                    # Direct path mode: infer name and reader from path
                    from pathlib import Path
                    path_obj = Path(direct_path)
                    name = entry.get("name", path_obj.name)
                    # Infer reader from path: look for known reader types in path
                    reader = self._infer_reader_from_direct_path(direct_path)
                    subtype = None
                    writer_conf = None  # No writer in direct mode
                else:
                    # Standard mode
                    name = entry["name"]
                    reader = entry.get("reader") or entry.get("reader_type") or default_readers.get(name)
                    subtype = entry.get("subtype", None)  # Deprecated: kept for backwards compat
                    writer_conf = entry.get("writers", entry.get("writer", default_writer))
            else:
                name = str(entry)
                reader = default_readers.get(name)
                subtype = None  # Inferred from path
                writer_conf = default_writer

            # Infer reader from path if not explicitly provided (standard mode only)
            if reader is None and direct_path is None:
                reader, actual_name = self._infer_reader(name)
                name = actual_name  # Update name to the extracted dataset name

            specs.append(
                DatasetSpec(
                    name=name,
                    reader=reader,
                    subtype=subtype,
                    writer_conf=writer_conf,
                    direct_path=direct_path,
                )
            )
        return specs

    def _prepare_dataset_workspace(
        self,
        spec: DatasetSpec,
        dataset_root: Path,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Create or refresh the per-dataset workspace within the experiment folder."""
        # Use experiment workspace directly, no per-dataset subdirectory
        dataset_workspace = self.base_workspace
        dataset_workspace.mkdir(parents=True, exist_ok=True)

        dataset_meta = self.metadata["datasets"].setdefault(
            spec.name,
            {
                "reader": spec.reader,
                "raw_source": str(dataset_root.resolve()),
                "workspace": str(dataset_workspace.resolve()),
                "ratios": {},
            },
        )
        dataset_meta["reader"] = spec.reader
        dataset_meta["raw_source"] = str(dataset_root.resolve())
        dataset_meta["workspace"] = str(dataset_workspace.resolve())

        return dataset_workspace, dataset_meta

    def _resolve_writer_plans(
        self,
        spec: DatasetSpec,
        override_writer: Optional[str] = None,
    ) -> List[WriterPlan]:
        """Instantiate writers based on dataset or global writer configuration.

        Args:
            spec: Dataset specification
            override_writer: Optional writer name to use instead of spec.writer_conf
                           (e.g., from reduction.writer or augmentation.writer)

        Returns:
            List of WriterPlan instances
        """
        # Use override writer if provided, otherwise use dataset writer_conf
        if override_writer is not None:
            conf = override_writer
        else:
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

            # Extract writer-specific parameters
            writer_kwargs = {}

            # Pass augmented_only_train to writers that support it (e.g., bert_int)
            if "augmented_only_train" in settings and writer_type.lower() == "bert_int":
                writer_kwargs["augmented_only_train"] = settings["augmented_only_train"]

            # Create writer with parameters
            try:
                writer = DatasetWriterFactory.create_writer(writer_type, **writer_kwargs)
            except TypeError as e:
                # Fallback: some writers don't accept kwargs, create without them
                if writer_kwargs:
                    logger.warning(f"Writer {writer_type} doesn't support kwargs {writer_kwargs}, creating without them")
                    writer = DatasetWriterFactory.create_writer(writer_type)
                else:
                    raise

            write_reduced = settings.get("write_reduced")
            if write_reduced is None:
                write_reduced = settings.get("write", True)
            if write_reduced is None:
                write_reduced = True

            write_augmented = settings.get("write_augmented")
            if write_augmented is None:
                write_augmented = settings.get("write", True)
            if write_augmented is None:
                write_augmented = True

            write_results = settings.get("write_results")
            if write_results is None:
                write_results = True

            plans.append(
                WriterPlan(
                    name=writer_type,
                    writer=writer,
                    write_reduced=bool(write_reduced),
                    write_augmented=bool(write_augmented),
                    write_results=bool(write_results),
                )
            )
        return plans

    def _is_experiment_complete(
        self,
        dataset_workspace: Path,
        artifact_root: Path,
        ratio_tag: str,
    ) -> bool:
        """Check if the experiment has been fully completed for this dataset and ratio.

        Returns True only if ALL required stages have been completed:
        - Reduction (if reduction_eval is true)
        - Augmentation (if augmentations exist and augmentation_eval is true)
        - Evaluation (for all specified models)
        """
        if not self.resume:
            return False

        # Check reduction results
        if self.reduction_eval:
            reduction_results = dataset_workspace / "reduction" / "results.json"
            if not reduction_results.exists():
                logger.debug(f"Experiment incomplete: reduction results missing at {reduction_results}")
                return False

        # Check augmentation results
        if self.augmentations and self.augmentation_eval:
            augmentation_results = dataset_workspace / "augmentation" / "results.json"
            if not augmentation_results.exists():
                logger.debug(f"Experiment incomplete: augmentation results missing at {augmentation_results}")
                return False

        # Check evaluation results - use unified results.json format
        evaluation_root = artifact_root / "evaluation"

        # Check baseline (reduction) evaluation
        if self.reduction_eval:
            baseline_results = evaluation_root / "baseline" / "results.json"
            if not baseline_results.exists():
                logger.debug(f"Experiment incomplete: baseline evaluation missing at {baseline_results}")
                return False

        # Check augmentation evaluations
        if self.augmentations:
            for aug_name in self.augmentations:
                aug_results = evaluation_root / aug_name / "results.json"
                if not aug_results.exists():
                    logger.debug(f"Experiment incomplete: augmentation evaluation missing for {aug_name} at {aug_results}")
                    return False

        logger.info(
            "✅ Experiment fully completed for ratio=%s — skipping execution",
            ratio_tag
        )
        return True

    def _check_early_completion(self) -> bool:
        """
        Fast check if entire experiment is already complete BEFORE loading anything.
        This is called at the very start of run() to avoid unnecessary dataset/model loading.

        Returns True if experiment should be skipped (already complete).
        """
        # Only skip if resume mode (i.e., not overwriting)
        if not self.resume:
            return False

        # Check if workspace exists at all
        if not self.base_workspace.exists():
            return False

        # For standard mode: check all ratios
        if not self.normalized_cfg.direct_mode:
            if not self.ratios:
                return False

            for ratio in self.ratios:
                ratio_tag = f"ratio_{ratio:.3f}".replace(".", "_")
                ratio_root = self.base_workspace / ratio_tag

                if not ratio_root.exists():
                    return False

                # Check each dataset in this ratio
                for dataset_dir in ratio_root.iterdir():
                    if not dataset_dir.is_dir():
                        continue

                    artifact_root = dataset_dir / "artifacts"
                    if not artifact_root.exists():
                        return False

                    # Use existing completion check logic
                    if not self._is_experiment_complete_early(dataset_dir, artifact_root):
                        return False
        else:
            # Direct mode: check single dataset workspace
            dataset_workspace = self.base_workspace
            artifact_root = dataset_workspace / "artifacts"

            if not artifact_root.exists():
                return False

            if not self._is_experiment_complete_early(dataset_workspace, artifact_root):
                return False

        return True

    def _check_early_completion_before_init(self) -> bool:
        """
        Ultra-fast check if experiment is already complete BEFORE initialization.
        Called in __init__ before creating workspace or loading datasets.

        Returns True if experiment should be skipped (already complete).
        """
        # Check if workspace exists at all
        if not self.base_workspace.exists():
            return False

        # Check if metadata file exists (basic indicator of previous run)
        if not self.metadata_file.exists():
            return False

        # For standard mode with ratios: quick check if ratio directories exist with results
        if self.ratios:
            for ratio in self.ratios:
                ratio_tag = f"ratio_{ratio:.3f}".replace(".", "_")
                ratio_root = self.base_workspace / ratio_tag

                if not ratio_root.exists():
                    return False

                # Look for any dataset directory with results
                has_complete_dataset = False
                for dataset_dir in ratio_root.iterdir():
                    if not dataset_dir.is_dir():
                        continue

                    # Quick check: look for results.json in expected locations
                    reduction_results = dataset_dir / "reduction" / "results.json"
                    if self.reduction_eval and not reduction_results.exists():
                        continue

                    if self.augmentations:
                        augmentation_results = dataset_dir / "augmentation" / "results.json"
                        if self.augmentation_eval and not augmentation_results.exists():
                            continue

                    # At least one complete dataset found
                    has_complete_dataset = True
                    break

                if not has_complete_dataset:
                    return False
        else:
            # Direct mode or no ratios: check main workspace
            reduction_results = self.base_workspace / "reduction" / "results.json"
            if self.reduction_eval and not reduction_results.exists():
                return False

            if self.augmentations:
                augmentation_results = self.base_workspace / "augmentation" / "results.json"
                if self.augmentation_eval and not augmentation_results.exists():
                    return False

        return True

    def _is_experiment_complete_early(
        self,
        dataset_workspace: Path,
        artifact_root: Path,
    ) -> bool:
        """
        Early completion check (used before loading datasets).
        Similar to _is_experiment_complete but doesn't need ratio_tag.
        """
        # Check reduction results
        if self.reduction_eval:
            reduction_results = dataset_workspace / "reduction" / "results.json"
            if not reduction_results.exists():
                return False

        # Check augmentation results
        if self.augmentations and self.augmentation_eval:
            augmentation_results = dataset_workspace / "augmentation" / "results.json"
            if not augmentation_results.exists():
                return False

        # Check evaluation results - use unified results.json format
        evaluation_root = artifact_root / "evaluation"

        # Check baseline evaluation (unified results.json for all models)
        if self.reduction_eval:
            baseline_results = evaluation_root / "baseline" / "results.json"
            if not baseline_results.exists():
                return False

        # Check augmentation evaluations (unified results.json for all models)
        if self.augmentations:
            for aug_name in self.augmentations:
                aug_results = evaluation_root / aug_name / "results.json"
                if not aug_results.exists():
                    return False

        return True

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

        # Copy seed if present (global configuration parameter)
        if "seed" in self.exp_cfg:
            cfg["seed"] = self.exp_cfg["seed"]

        # Lineage: essential metadata for models to determine context and paths
        lineage = cfg.setdefault("lineage", {})
        lineage["evaluation_root"] = str((ratio_root / "evaluation").resolve())
        lineage.setdefault("evaluation_dirs", {})
        lineage["raw_source"] = str(dataset_root.resolve())
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
        the reduction stage to create a reduced dataset. When ratio >= RATIO_THRESHOLD,
        it still executes the reduction stage to write the dataset workspace if a
        writer is configured (needed by models like bert_int and rrea), but without
        actually reducing the dataset.

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
        # Always execute reduction stage to ensure dataset workspace is written
        # (required by models like bert_int and rrea even when ratio=1.0)
        # The reduction stage will handle ratio >= 1.0 by not reducing the dataset
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

        return dataset_reduced

    def _execute_evaluations(
        self,
        dataset,
        dataset_reduced,
        reduction_stage,
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
        dataset_workspace: Path,
    ) -> None:
        """
        Execute baseline and augmentation evaluations.

        If no augmentation methods are configured, evaluates the baseline (reduced)
        dataset only. Otherwise, iterates through all configured augmentation methods,
        augments the dataset, and evaluates each augmented variant.

        Args:
            dataset: The original full dataset
            dataset_reduced: The reduced (or full) dataset to evaluate
            reduction_stage: ReductionStage instance (needed for retry mechanism)
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
            dataset_workspace: Root directory for dataset-level artefacts
        """
        reduction_meta = ratio_meta.get("reduction", {})
        reduction_results = Path(dataset_workspace / "reduction" / "results.json")

        # Proceed with normal evaluation (early skip is handled at ratio level)
        if self.reduction_eval:
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
            persisted = self._persist_stage_results(
                ratio_meta,
                VARIANT_BASELINE,
                reduction_results,
            )
            if persisted:
                reduction_meta["results"] = str(persisted)
        else:
            logger.info("⏭️  Skipping baseline evaluation (reduction.eval=false)")

        for aug_name in self.augmentations:
            # Check if auto-retry until improvement is enabled (from global config)
            retry_config = self.global_cfg.get("auto_retry_until_improvement", {})
            retry_enabled = retry_config.get("enabled", False)

            if retry_enabled and self.augmentation_eval and self.reduction_eval:
                # Check if augmentation results already exist (for resume)
                results_path = dataset_workspace / "augmentation" / "results.json"
                should_skip_retry = False

                if results_path.exists() and not self.overwrite_existing:
                    # Check if existing results meet the improvement threshold
                    metric = retry_config.get("metric", "hits@1")
                    min_improvement = retry_config.get("min_improvement", 0.01)

                    # Get reduction baseline
                    reduction_results_path = dataset_workspace / "reduction" / "results.json"
                    if reduction_results_path.exists():
                        with reduction_results_path.open("r") as f:
                            reduction_results = json.load(f)

                        baseline_value = None
                        for model_name, model_results in reduction_results.items():
                            if isinstance(model_results, dict) and metric in model_results:
                                baseline_value = model_results[metric]
                                break

                        if baseline_value is not None:
                            # Check existing augmentation results
                            with results_path.open("r") as f:
                                aug_results = json.load(f)

                            current_value = None
                            for model_name, model_results in aug_results.items():
                                if isinstance(model_results, dict) and metric in model_results:
                                    current_value = model_results[metric]
                                    break

                            if current_value is not None:
                                improvement = current_value - baseline_value
                                if improvement >= min_improvement:
                                    logger.info("✅ Skipping augmentation with retry '%s' (results already meet improvement threshold: %.4f >= %.4f)",
                                              aug_name, improvement, min_improvement)
                                    should_skip_retry = True
                                else:
                                    logger.info("🔁 Existing results insufficient (improvement: %.4f < %.4f), will retry augmentation '%s'",
                                              improvement, min_improvement, aug_name)

                if should_skip_retry:
                    augmentation_meta = ratio_meta.setdefault("augmentations", {}).setdefault(
                        aug_name, {}
                    )
                    augmentation_meta["results"] = str(results_path)
                    continue

                # Retry mechanism: repeat reduction + augmentation until improvement
                self._run_augmentation_with_retry(
                    reduction_stage,
                    augmentation_stage,
                    evaluation_stage,
                    aug_name,
                    dataset,
                    reader,
                    stage_cfg,
                    lineage,
                    ratio,
                    ratio_tag,
                    ratio_root,
                    ratio_meta,
                    spec,
                    dataset_workspace,
                    retry_config,
                )
            else:
                # Standard execution (no retry)
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

                # Apply filtering based on training_mode
                training_mode = stage_cfg.get("augmentation", {}).get("training_mode", "augmented")
                filtering_stage = FilteringStage(training_mode)
                dataset_filtered = filtering_stage.execute(dataset_reduced, dataset_augmented)

                if not self.augmentation_eval:
                    logger.info(
                        "⏭️  Skipping evaluation for augmentation '%s' (augmentation.eval=false)",
                        aug_name,
                    )
                    continue

                evaluation_stage.execute(
                    aug_name,
                    dataset_reduced,
                    dataset_filtered,
                    stage_cfg,
                    lineage,
                    ratio_root,
                    ratio_tag,
                    ratio_meta,
                )

                augmentation_meta = ratio_meta.setdefault("augmentations", {}).setdefault(
                    aug_name, {}
                )
                persisted = self._persist_stage_results(
                    ratio_meta,
                    aug_name,
                    Path(dataset_workspace / "augmentation" / "results.json"),
                )
                if persisted:
                    augmentation_meta["results"] = str(persisted)

    def _run_augmentation_with_retry(
        self,
        reduction_stage,
        augmentation_stage,
        evaluation_stage,
        aug_name: str,
        dataset,
        reader,
        stage_cfg: Dict[str, Any],
        lineage: Dict[str, Any],
        ratio: float,
        ratio_tag: str,
        ratio_root: Path,
        ratio_meta: Dict[str, Any],
        spec,
        dataset_workspace: Path,
        retry_config: Dict[str, Any],
    ) -> None:
        """Run reduction + augmentation with retry mechanism until improvement or max attempts.

        Each attempt:
        1. Creates new reduction (with different seed)
        2. Evaluates the reduction baseline
        3. Applies augmentation to new reduction
        4. Evaluates augmentation
        5. Compares augmentation vs. NEW reduction baseline
        """
        max_attempts = retry_config.get("max_attempts", 5)
        metric = retry_config.get("metric", "hits@1")
        min_improvement = retry_config.get("min_improvement", 0.01)
        save_all = retry_config.get("save_all_attempts", True)

        logger.info("="*80)
        logger.info("🔁 AUTO-RETRY ENABLED: Will retry reduction+augmentation until improvement")
        logger.info(f"   Target metric: {metric}")
        logger.info(f"   Min improvement: {min_improvement:.4f}")
        logger.info(f"   Max attempts: {max_attempts}")
        logger.info("="*80)

        best_value = -1.0
        best_attempt = 0
        best_results_file = None
        best_baseline_value = None

        # Save original resume flag and disable caching during retry
        original_resume = reduction_stage.resume
        reduction_stage.resume = False
        augmentation_stage.resume = False

        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 ATTEMPT {attempt}/{max_attempts}")
            logger.info(f"{'='*80}")

            # Step 1: Create NEW reduction for this attempt
            # Skip writing datasets ONLY if we don't need to save them later
            skip_write = not self.reduction_save_dataset and not self.reduction_save_model
            logger.info(f"🔨 Running reduction (skip_write={skip_write})...")
            if ratio < 1.0:
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
                    skip_dataset_write=skip_write,
                )
            else:
                dataset_reduced = dataset.clone()

            # Step 2: Evaluate NEW reduction baseline
            logger.info(f"📊 Evaluating reduction baseline...")
            evaluation_stage.execute(
                "baseline",
                dataset_reduced,
                None,
                stage_cfg,
                lineage,
                ratio_root,
                ratio_tag,
                ratio_meta,
            )

            # Get baseline value for THIS reduction
            reduction_results_path = dataset_workspace / "reduction" / "results.json"
            if not reduction_results_path.exists():
                logger.warning(f"⚠️  Reduction results not found for attempt {attempt}, skipping")
                continue

            with reduction_results_path.open("r") as f:
                reduction_results = json.load(f)

            baseline_value = None
            for model_name, model_results in reduction_results.items():
                if isinstance(model_results, dict) and metric in model_results:
                    baseline_value = model_results[metric]
                    break

            if baseline_value is None:
                logger.warning(f"⚠️  Metric '{metric}' not found in reduction results for attempt {attempt}")
                continue

            logger.info(f"   Reduction baseline ({metric}): {baseline_value:.4f}")

            # Step 3: Run augmentation on NEW reduction
            # Skip writing datasets ONLY if we don't need to save them later
            skip_write_aug = not self.augmentation_save_dataset and not self.augmentation_save_model
            logger.info(f"✨ Running augmentation (skip_write={skip_write_aug})...")
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
                skip_dataset_write=skip_write_aug,
            )

            # Apply filtering based on training_mode
            training_mode = stage_cfg.get("augmentation", {}).get("training_mode", "augmented")
            filtering_stage = FilteringStage(training_mode)
            dataset_filtered = filtering_stage.execute(dataset_reduced, dataset_augmented)

            # Step 4: Evaluate augmentation
            logger.info(f"📊 Evaluating augmentation...")
            evaluation_stage.execute(
                aug_name,
                dataset_reduced,
                dataset_filtered,
                stage_cfg,
                lineage,
                ratio_root,
                ratio_tag,
                ratio_meta,
            )

            # Get results
            augmentation_meta = ratio_meta.setdefault("augmentations", {}).setdefault(
                aug_name, {}
            )
            results_path = dataset_workspace / "augmentation" / "results.json"
            persisted = self._persist_stage_results(
                ratio_meta,
                aug_name,
                results_path,
            )

            if persisted and persisted.exists():
                with persisted.open("r") as f:
                    aug_results = json.load(f)

                # Extract metric value
                current_value = None
                for model_name, model_results in aug_results.items():
                    if isinstance(model_results, dict) and metric in model_results:
                        current_value = model_results[metric]
                        break

                if current_value is not None:
                    improvement = current_value - baseline_value
                    logger.info(f"📊 Results - Attempt {attempt}:")
                    logger.info(f"   Baseline ({metric}): {baseline_value:.4f}")
                    logger.info(f"   Augmented ({metric}): {current_value:.4f}")
                    logger.info(f"   Improvement: {improvement:+.4f} ({improvement/baseline_value*100:+.2f}%)")

                    # Save attempt results if configured
                    if save_all:
                        attempt_file = results_path.parent / f"results_attempt_{attempt}.json"
                        shutil.copy(persisted, attempt_file)
                        logger.info(f"💾 Saved attempt {attempt} → {attempt_file}")

                    # Check if improvement is sufficient
                    if improvement >= min_improvement:
                        logger.info(f"✅ SUCCESS: Improvement >= {min_improvement:.4f}!")
                        logger.info(f"   Best result achieved at attempt {attempt}")

                        # Restore resume flags
                        reduction_stage.resume = original_resume
                        augmentation_stage.resume = original_resume

                        augmentation_meta["results"] = str(persisted)
                        augmentation_meta["retry_attempts"] = attempt
                        augmentation_meta["retry_improvement"] = improvement
                        augmentation_meta["retry_baseline"] = baseline_value
                        return
                    elif current_value > best_value:
                        best_value = current_value
                        best_attempt = attempt
                        best_results_file = persisted
                        best_baseline_value = baseline_value
                        logger.info(f"   🏆 New best value: {best_value:.4f}")
                else:
                    logger.warning(f"⚠️  Could not extract metric '{metric}' from attempt {attempt}")
            else:
                logger.warning(f"⚠️  Results file not found for attempt {attempt}")

        # Restore resume flags
        reduction_stage.resume = original_resume
        augmentation_stage.resume = original_resume

        # Max attempts reached without sufficient improvement
        logger.warning(f"⚠️  Max attempts ({max_attempts}) reached without sufficient improvement")

        if best_attempt > 0 and best_baseline_value is not None:
            improvement = best_value - best_baseline_value
            logger.info(f"   Best value: {best_value:.4f} (attempt {best_attempt})")
            logger.info(f"   Best baseline: {best_baseline_value:.4f}")
            logger.info(f"   Best improvement: {improvement:+.4f}")

            # Use best attempt results (not last attempt!)
            if best_results_file and best_results_file.exists():
                # Copy best attempt to main results.json
                results_path = dataset_workspace / "augmentation" / "results.json"
                shutil.copy(best_results_file, results_path)
                logger.info(f"✅ Using best attempt {best_attempt} → {results_path}")

                augmentation_meta["results"] = str(results_path)
                augmentation_meta["retry_attempts"] = max_attempts
                augmentation_meta["retry_improvement"] = improvement
                augmentation_meta["retry_best_attempt"] = best_attempt
                augmentation_meta["retry_baseline"] = best_baseline_value
            else:
                logger.warning("⚠️  Best attempt results file not found")
        else:
            logger.warning("⚠️  No valid attempts completed")

    def _persist_stage_results(
        self,
        ratio_meta: Dict[str, Any],
        variant_name: Optional[str],
        target_file: Path,
    ) -> Optional[Path]:
        """Results are now saved directly by EvaluationStage, so this just checks if they exist."""
        variant_key = EvaluationStage._normalise_variant_key(variant_name)
        evaluation_meta = ratio_meta.get("evaluations", {}).get(variant_key)
        if not evaluation_meta:
            return None

        # Check if results.json exists in the stage directory
        results_path = evaluation_meta.get("paths", {}).get("results")
        if results_path and Path(results_path).exists():
            logger.info("💾 Results already saved → %s", results_path)
            return Path(results_path)

        return None

    def _collect_model_results(self, evaluation_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dictionary with model scores loaded from their JSON files."""
        aggregated: Dict[str, Any] = {}
        for model_name, file_path in evaluation_meta.get("paths", {}).items():
            path_obj = Path(file_path)
            if not path_obj.exists():
                continue
            try:
                with path_obj.open("r", encoding="utf-8") as handle:
                    aggregated[model_name] = json.load(handle)
            except json.JSONDecodeError:
                logger.warning("Failed to parse evaluation results from %s", path_obj)
        return aggregated

    def _build_metadata_payload(self) -> Dict[str, Any]:
        """Return a flattened metadata dictionary for persistence."""
        dataset_entry = next(iter(self.metadata.get("datasets", {}).items()), (None, {}))
        dataset_name, dataset_meta = dataset_entry
        ratio_meta = next(iter(dataset_meta.get("ratios", {}).values()), {})

        augmentation_section = None
        augmentations_dict = ratio_meta.get("augmentations", {})
        if augmentations_dict:
            aug_name, aug_meta = next(iter(augmentations_dict.items()))
            augmentation_section = {
                "name": aug_name,
                "details": copy.deepcopy(aug_meta),
            }

        payload = {
            "name": self.name,
            "dataset": dataset_name,
            "reader": dataset_meta.get("reader"),
            "raw_source": dataset_meta.get("raw_source"),
            "workspace": dataset_meta.get("workspace"),
            "artifact_root": dataset_meta.get("artifact_root"),
            "reduction_method": self.reduction_method,
            "ratio": self.ratio_value,
            "reduction": copy.deepcopy(ratio_meta.get("reduction")),
            "augmentation": augmentation_section,
            "evaluations": copy.deepcopy(ratio_meta.get("evaluations")),
            "augmentation_method": self.augmentations[0] if self.augmentations else None,
            "models": self.models,
            "overwrite_existing": self.overwrite_existing,
            "workspace_root": str(self.base_workspace.resolve()),
        }
        return payload

    def _write_metadata(self) -> None:
        """Persist the experiment metadata summary alongside artefacts."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_metadata_payload()
        with self.metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("🗒️  Experiment metadata saved → %s", self.metadata_file)

    def _cleanup_intermediate_files(self) -> None:
        """Remove intermediate files from current experiment folder."""
        logger.info("🧹 Cleaning intermediate files from experiment: %s", self.name)
        metadata = self._load_metadata_snapshot()
        target_dirs = self._collect_intermediate_dirs(metadata)

        removed = 0
        failed = 0
        reclaimed_bytes = 0
        for path in sorted(target_dirs):
            dir_path = Path(path)
            if not dir_path.exists():
                continue
            if not self._is_within_workspace(dir_path):
                logger.debug("Skipping cleanup for path outside workspace: %s", dir_path)
                continue

            dir_size = self._directory_size(dir_path)
            try:
                shutil.rmtree(dir_path, ignore_errors=False)
                # Verify directory was actually removed
                if not dir_path.exists():
                    reclaimed_bytes += dir_size
                    removed += 1
                    logger.debug("✓ Removed: %s", dir_path)
                else:
                    logger.warning("⚠ Failed to remove (still exists): %s", dir_path)
                    failed += 1
            except Exception as e:
                logger.warning("⚠ Failed to remove %s: %s", dir_path, e)
                failed += 1

        if removed:
            logger.info(
                "🧹 Removed %d artefact directories, freed %.1f MB",
                removed,
                reclaimed_bytes / (1024 * 1024),
            )
        if failed:
            logger.warning(
                "⚠ Failed to remove %d directories (check permissions or open files)",
                failed,
            )
        if not removed and not failed:
            logger.info("🧹 No intermediate artefacts detected for cleanup.")

    def _load_metadata_snapshot(self) -> Dict[str, Any]:
        """Return the latest metadata dictionary, reading from disk if needed."""
        if self.metadata.get("datasets"):
            return self.metadata
        if self.metadata_file.exists():
            with self.metadata_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return self.metadata

    def _collect_intermediate_dirs(self, metadata: Dict[str, Any]) -> List[str]:
        """Collect directories to remove with clear: true.

        Respects save_dataset and save_model flags - only deletes artifacts
        that are not explicitly saved.

        Searches directly in the filesystem to find artifacts, making this robust
        to incomplete or stale metadata.
        """
        targets: List[str] = []

        # Search directly in the workspace directory
        workspace = self.base_workspace

        # If all flags are false, mark all stage directories for removal
        all_flags_false = (
            not self.reduction_save_dataset
            and not self.reduction_save_model
            and not self.augmentation_save_dataset
            and not self.augmentation_save_model
        )

        if all_flags_false:
            # Remove artifact root
            artifact_root = workspace / "artifact"
            if artifact_root.exists():
                targets.append(str(artifact_root))
                logger.debug("Marked for cleanup: %s (all save flags=false)", artifact_root)

            # Remove reduction datasets and models (but keep reduction/ directory itself)
            # IMPORTANT: We only remove subdirectories (dataset/, model/)
            # The reduction/ directory must remain to preserve summary.json and results.json
            reduction_dir = workspace / "reduction"
            if reduction_dir.exists():
                for subdir in ["dataset", "model"]:
                    path = reduction_dir / subdir
                    if path.exists():
                        targets.append(str(path))
                        logger.debug("Marked for cleanup: %s (all save flags=false)", path)

            # Remove augmentation datasets and models (but keep augmentation/ directory itself)
            # IMPORTANT: We only remove subdirectories (dataset/, model/)
            # The augmentation/ directory must remain to preserve summary.json and results.json
            augmentation_dir = workspace / "augmentation"
            if augmentation_dir.exists():
                for subdir in ["dataset", "model"]:
                    path = augmentation_dir / subdir
                    if path.exists():
                        targets.append(str(path))
                        logger.debug("Marked for cleanup: %s (all save flags=false)", path)

            return targets

        # Otherwise, collect specific subdirectories based on flags
        # IMPORTANT: We only remove subdirectories (dataset/, model/), never the stage
        # directories themselves (reduction/, augmentation/) because they contain
        # summary.json and results.json which must be preserved.

        # Check reduction artifacts
        reduction_dir = workspace / "reduction"
        if reduction_dir.exists():
            # Check for dataset subdirectory
            dataset_dir = reduction_dir / "dataset"
            if dataset_dir.exists() and not self.reduction_save_dataset:
                targets.append(str(dataset_dir))
                logger.debug("Marked for cleanup: %s (save_dataset=false)", dataset_dir)

            # Check for model subdirectory
            model_dir = reduction_dir / "model"
            if model_dir.exists() and not self.reduction_save_model:
                targets.append(str(model_dir))
                logger.debug("Marked for cleanup: %s (save_model=false)", model_dir)

        # Check augmentation artifacts
        augmentation_dir = workspace / "augmentation"
        if augmentation_dir.exists():
            # Check for dataset subdirectory
            dataset_dir = augmentation_dir / "dataset"
            if dataset_dir.exists() and not self.augmentation_save_dataset:
                targets.append(str(dataset_dir))
                logger.debug("Marked for cleanup: %s (save_dataset=false)", dataset_dir)

            # Check for model subdirectory
            model_dir = augmentation_dir / "model"
            if model_dir.exists() and not self.augmentation_save_model:
                targets.append(str(model_dir))
                logger.debug("Marked for cleanup: %s (save_model=false)", model_dir)

        return targets

    def _schedule_stage_cleanup(self) -> List[Path]:
        """Determine which stage artefacts should be deleted after the run.

        Uses granular save_dataset and save_model flags to selectively delete
        only the artifacts that should not be saved.
        """
        pending: List[Path] = []

        for dataset_meta in self.metadata.get("datasets", {}).values():
            for ratio_meta in dataset_meta.get("ratios", {}).values():
                # Handle reduction cleanup with granular flags
                reduction_meta = ratio_meta.get("reduction")
                if reduction_meta:
                    pending.extend(
                        self._extract_stage_paths_granular(
                            reduction_meta,
                            save_dataset=self.reduction_save_dataset,
                            save_model=self.reduction_save_model
                        )
                    )

                # Handle augmentation cleanup with granular flags
                for augmentation_meta in ratio_meta.get("augmentations", {}).values():
                    pending.extend(
                        self._extract_stage_paths_granular(
                            augmentation_meta,
                            save_dataset=self.augmentation_save_dataset,
                            save_model=self.augmentation_save_model
                        )
                    )

        return pending

    @staticmethod
    def _extract_stage_paths(
        stage_meta: Optional[Dict[str, Any]], *, delete_parent: bool = False
    ) -> List[Path]:
        if not stage_meta:
            return []
        path_dict = stage_meta.get("paths", {})
        collected = [Path(path) for path in path_dict.values()]
        stage_meta["paths"] = {}
        # NOTE: We no longer delete the parent directory when delete_parent=True
        # because the parent (e.g., augmentation/) contains results.json and summary.json
        # which must be preserved even when save=false.
        # Instead, we only delete the specific subdirectories (dataset/, model/, etc.)
        return collected

    @staticmethod
    def _extract_stage_paths_granular(
        stage_meta: Optional[Dict[str, Any]],
        *,
        save_dataset: bool,
        save_model: bool
    ) -> List[Path]:
        """Extract paths to delete based on granular save flags.

        Args:
            stage_meta: Metadata for the stage (reduction or augmentation)
            save_dataset: Whether to preserve the dataset
            save_model: Whether to preserve the model

        Returns:
            List of paths to delete
        """
        if not stage_meta:
            return []

        path_dict = stage_meta.get("paths", {})
        paths_to_delete: List[Path] = []

        # Selectively delete based on flags
        for key, path_str in list(path_dict.items()):
            path = Path(path_str)

            # Check if this is a dataset or model path
            if "dataset" in key.lower():
                if not save_dataset:
                    paths_to_delete.append(path)
                    del path_dict[key]
            elif "model" in key.lower():
                if not save_model:
                    paths_to_delete.append(path)
                    del path_dict[key]
            else:
                # For other paths (e.g., "output"), delete if neither flag is set
                if not save_dataset and not save_model:
                    paths_to_delete.append(path)
                    del path_dict[key]

        return paths_to_delete

    @staticmethod
    def _remove_stage_outputs(paths: List[Path]) -> None:
        for path in paths:
            if not path.exists():
                continue
            try:
                shutil.rmtree(path)
                logger.info("🧽 Removed stage artefacts at %s (save=false)", path)
            except OSError as exc:
                logger.warning("Failed to remove artefacts at %s: %s", path, exc)

    def _is_within_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.base_workspace.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    continue
        return total
