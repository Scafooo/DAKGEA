"""Core experiment orchestration logic."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.loader import Config
from src.core.dataset import Dataset
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory
from src.logger import get_logger

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
    ReductionStage,
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

        global_cfg = Config().get()
        self.paths = global_cfg["paths"]
        default_overwrite = global_cfg.get("experiment_defaults", {}).get(
            "overwrite_existing", False
        )

        self.normalized_cfg = ExperimentConfig.from_payload(
            exp_cfg,
            cli_overwrite=overwrite_existing,
            default_overwrite=default_overwrite,
        )
        self.name = self.normalized_cfg.name
        self.datasets_cfg = list(self.normalized_cfg.datasets)
        self.ratios = list(self.normalized_cfg.ratios)
        self.augmentations = list(self.normalized_cfg.augmentations)
        self.models = list(self.normalized_cfg.models)
        self.reduction_method = self.normalized_cfg.reduction_method
        self.reduction_writer = self.normalized_cfg.reduction_writer
        self.reduction_save = self.normalized_cfg.reduction_save
        self.reduction_eval = self.normalized_cfg.reduction_eval
        self.augmentation_writer = self.normalized_cfg.augmentation_writer
        self.augmentation_save = self.normalized_cfg.augmentation_save
        self.augmentation_eval = self.normalized_cfg.augmentation_eval
        self.clear_intermediate = self.normalized_cfg.clear_intermediate
        self.overwrite_existing = self.normalized_cfg.overwrite_existing
        self.resume = self.normalized_cfg.resume

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
        # Check if we're in direct path mode (no ratios = direct dataset access)
        direct_mode = self.normalized_cfg.direct_mode

        if direct_mode:
            self._run_direct_mode()
        else:
            self._run_standard_mode()
        
        # Clean up intermediate files if requested
        if self.clear_intermediate:
            self._cleanup_intermediate_files()

    def _run_direct_mode(self) -> None:
        """Run experiments in direct path mode: read datasets directly, skip reduction/writer."""
        logger.info(
            "=== Running in DIRECT PATH mode: '%s' ===",
            self.name,
        )
        logger.info("Skipping reduction, augmentation, and writer stages")

        progress = ProgressTracker(total=len(self.datasets), enabled=self.show_progress)

        try:
            for spec in self.datasets:
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
                reader = DatasetReaderFactory.create_reader(spec.reader)
                dataset = reader.read(str(dataset_root))

                ratio_tag = "direct"
                ratio = 1.0
                ratio_root = dataset_workspace

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
                lineage["direct_mode"] = True
                lineage["dataset_workspace"] = str(dataset_root.resolve())
                lineage["raw_source"] = str(dataset_root.resolve())

                ratio_meta = dataset_meta["ratios"].setdefault(ratio_tag, {})
                ratio_meta.update(
                    {
                        "ratio": ratio,
                        "target_entities": stage_cfg["reduction"]["target_entities"],
                    }
                )
                ratio_meta.setdefault(
                    "reduction", {"method": "direct", "paths": {}}
                )
                ratio_meta.setdefault("augmentations", {})
                ratio_meta.setdefault("evaluations", {})

                evaluation_stage = EvaluationStage(
                    self._resolve_writer_plans(spec),
                    self.models,
                    self.resume,
                )
                logger.info("[STEP] Running evaluation for '%s' (direct path)", spec.name)

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
                progress.step()
        finally:
            progress.close()
            self._write_metadata()
            logger.info("=== Direct mode experiments completed ===")

    def _run_standard_mode(self) -> None:
        """Run experiments in standard mode with reduction, augmentation, and writer."""
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
                dataset = reader.read(str(dataset_root))

                # Resolve writer plans for each stage
                # Reduction stage: use reduction.writer if specified, else dataset.writer
                reduction_writer_plans = self._resolve_writer_plans(
                    spec, override_writer=self.reduction_writer
                )
                # Augmentation stage: use augmentation.writer if specified, else dataset.writer
                augmentation_writer_plans = self._resolve_writer_plans(
                    spec, override_writer=self.augmentation_writer
                )
                # Evaluation stage: always use dataset.writer (no phase-specific override)
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

    def _resolve_writer_plans(
        self, spec: DatasetSpec, override_writer: Optional[str] = None
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
        else:
            # No reduction needed - use full dataset
            dataset_reduced = dataset.clone()
            reduction_meta = ratio_meta.setdefault(
                "reduction",
                {"method": "none", "paths": {}},
            )
            reduction_meta["method"] = "none"

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

    def _write_metadata(self) -> None:
        """Persist the experiment metadata summary alongside artefacts."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)
        logger.info("🗒️  Experiment metadata saved → %s", self.metadata_file)

    def _cleanup_intermediate_files(self) -> None:
        """Remove intermediate files from current experiment folder."""
        logger.info("🧹 Cleaning intermediate files from experiment: %s", self.name)
        metadata = self._load_metadata_snapshot()
        target_dirs = self._collect_intermediate_dirs(metadata)

        removed = 0
        reclaimed_bytes = 0
        for path in sorted(target_dirs):
            dir_path = Path(path)
            if not dir_path.exists():
                continue
            if not self._is_within_workspace(dir_path):
                logger.debug("Skipping cleanup for path outside workspace: %s", dir_path)
                continue
            reclaimed_bytes += self._directory_size(dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)
            removed += 1

        if removed:
            logger.info(
                "🧹 Removed %d artefact directories, freed %.1f MB",
                removed,
                reclaimed_bytes / (1024 * 1024),
            )
        else:
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
        """Collect directories produced during reduction/augmentation stages."""
        targets: List[str] = []
        for dataset_meta in metadata.get("datasets", {}).values():
            for ratio_meta in dataset_meta.get("ratios", {}).values():
                reduction_paths = ratio_meta.get("reduction", {}).get("paths", {})
                targets.extend(reduction_paths.values())
                for augmentation_meta in ratio_meta.get("augmentations", {}).values():
                    targets.extend(augmentation_meta.get("paths", {}).values())
        return targets

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
