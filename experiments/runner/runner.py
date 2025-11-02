"""Core experiment orchestration logic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config.loader import Config
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory
from src.logger import get_logger

logger = get_logger(__name__)

from .progress import ProgressTracker
from .specs import DatasetSpec, WriterPlan
from .stages import AugmentationStage, EvaluationStage, ReductionStage


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
            self.datasets_cfg = exp_cfg["datasets"]
            self.ratios = [float(r) for r in exp_cfg["reduction_ratios"]]
            self.augmentations = exp_cfg.get("augmentation_methods", [])
            self.models = exp_cfg.get("models_to_run", [])
            self.reduction_method = exp_cfg.get("reduction_method", "ids")
        except KeyError as exc:
            raise KeyError(f"Missing required experiment configuration key: {exc}") from exc

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
        for candidate in self.base_data.iterdir():
            if not candidate.is_dir():
                continue
            if (candidate / dataset_name).is_dir():
                logger.debug(
                    "Auto-detected reader '%s' for dataset '%s'",
                    candidate.name,
                    dataset_name,
                )
                return candidate.name
        raise FileNotFoundError(
            f"Unable to infer reader for dataset '{dataset_name}'. "
            f"Expected to find it under {self.base_data}."
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

                for ratio in self.ratios:
                    ratio_desc = f"{ratio * 100:.1f}%"
                    ratio_tag = self._format_ratio_tag(ratio)
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

                    if not self.augmentations:
                        evaluation_stage.execute(
                            "baseline",
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
        lineage = cfg.setdefault("lineage", {})
        lineage["reduction_method"] = self.reduction_method
        lineage["dataset_workspace"] = str(dataset_workspace.resolve())
        lineage["ratio_tag"] = ratio_tag
        lineage["ratio_root"] = str(ratio_root.resolve())
        reduction_root = ratio_root / "reduction"
        reduction_artefacts = reduction_root / "artefacts"
        augmentation_root = ratio_root / "augmentation"
        evaluation_root = ratio_root / "evaluation"
        lineage["reduction_root"] = str(reduction_root.resolve())
        lineage["reduction_artefacts"] = str(reduction_artefacts.resolve())
        lineage["reduced_base"] = str(reduction_artefacts.resolve())
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

    def _write_metadata(self) -> None:
        """Persist the experiment metadata summary alongside artefacts."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)
        logger.info("🗒️  Experiment metadata saved → %s", self.metadata_file)
