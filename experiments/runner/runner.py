"""Core experiment orchestration logic."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.alignment_models.registry import MODEL_REGISTRY
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.config.loader import Config
from src.core.dataset import Dataset
from src.core.dataset.reader import ReaderFactory
from src.core.dataset.writer import WriterFactory
from src.logger import get_logger

logger = get_logger(__name__)
from src.reduction.registry import REDUCTION_REGISTRY

from .progress import ProgressTracker
from .specs import DatasetSpec, WriterPlan


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
        self.base_reduced: Path = Path(self.paths["reduced_data"])
        self.base_augmented: Path = Path(self.paths["augmented_data"])
        self.base_output: Path = Path(self.paths["results"]) / self.name
        self.base_output.mkdir(parents=True, exist_ok=True)

        self.datasets: List[DatasetSpec] = self._build_dataset_specs()
        self.reducer_cls = REDUCTION_REGISTRY.get(self.reduction_method)

    @staticmethod
    def _has_cached(path: Path) -> bool:
        """Return True when a directory exists and contains files."""
        return path.exists() and any(path.iterdir())

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
                dataset_out_dir = self.base_output / spec.name
                dataset_out_dir.mkdir(parents=True, exist_ok=True)

                reader = ReaderFactory.create_reader(spec.reader)
                dataset = reader.read(str(dataset_root), subtype=spec.subtype)
                writer_plans = self._resolve_writer_plans(spec)

                logger.info(
                    "→ Dataset '%s' (reader=%s, subtype=%s)",
                    spec.name,
                    spec.reader,
                    spec.subtype,
                )
                logger.info("[STEP] Preparing dataset '%s'", spec.name)

                for ratio in self.ratios:
                    ratio_desc = f"{ratio * 100:.1f}%"
                    progress.set_description(f"📦 {spec.name} [{ratio_desc}]")
                    logger.info("[STEP] Ratio %.1f%% for dataset '%s'", ratio * 100, spec.name)
                    stage_cfg = self._build_stage_config(
                        spec.name, len(dataset.aligned_entities), ratio
                    )
                    dataset_reduced = self._perform_reduction(
                        reader, writer_plans, dataset, spec, ratio, stage_cfg
                    )

                    if not self.augmentations:
                        self._evaluate_models(
                            writer_plans,
                            dataset_reduced,
                            dataset_reduced,
                            spec,
                            ratio,
                            stage_cfg,
                            "baseline",
                        )

                    for aug_name in self.augmentations:
                        dataset_augmented = self._perform_augmentation(
                            reader,
                            writer_plans,
                            dataset_reduced,
                            spec,
                            ratio,
                            stage_cfg,
                            aug_name,
                        )
                        self._evaluate_models(
                            writer_plans,
                            dataset_reduced,
                            dataset_augmented,
                            spec,
                            ratio,
                            stage_cfg,
                            aug_name,
                        )
                    progress.step()
        finally:
            progress.close()
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

            writer = WriterFactory.create_writer(writer_type)
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
                "ratio_tag": self._format_ratio_tag(ratio),
            }
        )
        lineage = cfg.setdefault("lineage", {})
        lineage.setdefault("reduction_method", self.reduction_method)
        lineage.setdefault(
            "reduced_base",
            str((self.base_reduced / self.reduction_method).resolve()),
        )
        lineage.setdefault(
            "augmented_base",
            str((self.base_augmented / self.reduction_method).resolve()),
        )
        lineage.setdefault("reduced_paths", {})
        lineage.setdefault("augmented_paths", {})
        lineage.setdefault("augmented_hybea_paths", {})
        lineage.setdefault("ratio_tag", experiment_meta["ratio_tag"])
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

    @staticmethod
    def _select_reader_plan(plans: List[WriterPlan], preferred: str) -> str:
        """Choose which writer plan directory should be used to reload cached data."""
        for plan in plans:
            if plan.name == preferred:
                return plan.name
        return plans[0].name if plans else preferred

    def _perform_reduction(
        self,
        reader,
        writer_plans: List[WriterPlan],
        dataset: Dataset,
        spec: DatasetSpec,
        ratio: float,
        stage_cfg: Dict[str, Any],
    ) -> Dataset:
        """Reduce the dataset or reuse cached artefacts when resuming."""
        ratio_tag = self._format_ratio_tag(ratio)
        logger.info("[STEP] Reduction stage → dataset=%s ratio=%s", spec.name, ratio_tag)
        reader_plan = self._select_reader_plan(writer_plans, spec.reader)
        reader_root = (
            self.base_reduced
            / self.reduction_method
            / reader_plan
            / spec.name
            / ratio_tag
        )

        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping reduction for %s (%s) — cached artefacts detected",
                spec.name,
                ratio_tag,
            )
            try:
                return reader.read(str(reader_root), subtype=spec.subtype)
            except FileNotFoundError:
                logger.warning(
                    "Cached reduction for %s (%s) is incomplete; recomputing.",
                    spec.name,
                    ratio_tag,
                )

        logger.debug("[IMPORTANT] Instantiating reducer '%s'", self.reduction_method)
        reducer = self.reducer_cls(stage_cfg)
        dataset_reduced = reducer.reduce(dataset.clone())
        logger.info("[SUCCESS] Reduction complete (%d aligned pairs)", len(dataset_reduced.aligned_entities))

        lineage = stage_cfg.setdefault("lineage", {})
        reduced_paths = lineage.setdefault("reduced_paths", {})

        for plan in writer_plans:
            plan_root = (
                self.base_reduced
                / self.reduction_method
                / plan.name
                / spec.name
                / ratio_tag
            )
            reduced_paths[plan.name] = str(plan_root)
            if getattr(plan.writer, "file_type", None) == "hybea" or plan.name == "hybea":
                lineage["reduced_hybea_path"] = str(plan_root)

            if plan.write_reduced:
                plan.writer.write(dataset_reduced, str(plan_root))
                logger.info("📝 [%s] Saved reduced dataset → %s", plan.name, plan_root)
            else:
                logger.debug("Skipping reduced write for plan '%s' (write_reduced=False)", plan.name)

        return dataset_reduced

    def _perform_augmentation(
        self,
        reader,
        writer_plans: List[WriterPlan],
        dataset_reduced: Dataset,
        spec: DatasetSpec,
        ratio: float,
        stage_cfg: Dict[str, Any],
        augmentation_name: str,
    ) -> Dataset:
        """Augment the reduced dataset or reuse cached artefacts when resuming."""
        ratio_tag = self._format_ratio_tag(ratio)
        logger.info("[STEP] Augmentation stage → %s (ratio=%s)", augmentation_name, ratio_tag)
        reader_plan = self._select_reader_plan(writer_plans, spec.reader)
        reader_root = (
            self.base_augmented
            / self.reduction_method
            / augmentation_name
            / reader_plan
            / spec.name
            / ratio_tag
        )
        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping augmentation '%s' for %s (%s) — cached artefacts detected",
                augmentation_name,
                spec.name,
                ratio_tag,
            )
            try:
                return reader.read(str(reader_root), subtype=spec.subtype)
            except FileNotFoundError:
                logger.warning(
                    "Cached augmentation '%s' for %s (%s) is incomplete; recomputing.",
                    augmentation_name,
                    spec.name,
                    ratio_tag,
                )

        augmenter_cls = AUGMENTATION_REGISTRY.get(augmentation_name)
        logger.debug("[IMPORTANT] Instantiating augmenter '%s'", augmentation_name)
        augmenter = augmenter_cls(stage_cfg)
        dataset_augmented = augmenter.augment(dataset_reduced.clone())
        logger.info("[SUCCESS] Augmentation '%s' complete (%d aligned pairs)", augmentation_name, len(dataset_augmented.aligned_entities))

        lineage = stage_cfg.setdefault("lineage", {})
        augmented_paths = lineage.setdefault("augmented_paths", {})
        per_aug_paths = augmented_paths.setdefault(augmentation_name, {})
        hybea_aug_paths = lineage.setdefault("augmented_hybea_paths", {})

        for plan in writer_plans:
            plan_root = (
                self.base_augmented
                / self.reduction_method
                / augmentation_name
                / plan.name
                / spec.name
                / ratio_tag
            )
            per_aug_paths[plan.name] = str(plan_root)
            if getattr(plan.writer, "file_type", None) == "hybea" or plan.name == "hybea":
                hybea_aug_paths[augmentation_name] = str(plan_root)

            if plan.write_augmented:
                plan.writer.write(dataset_augmented, str(plan_root))
                logger.info("📝 [%s] Saved augmented dataset → %s", plan.name, plan_root)
            else:
                logger.debug("Skipping augmented write for plan '%s' (write_augmented=False)", plan.name)

        return dataset_augmented

    def _evaluate_models(
        self,
        writer_plans: List[WriterPlan],
        dataset_reduced: Dataset,
        dataset_augmented: Dataset,
        spec: DatasetSpec,
        ratio: float,
        stage_cfg: Dict[str, Any],
        augmentation_name: str,
    ) -> None:
        """Evaluate all configured models and persist results when requested."""
        ratio_tag = self._format_ratio_tag(ratio)
        logger.info("[STEP] Evaluation stage → augmentation=%s", augmentation_name)
        results_dir = self.base_output / spec.name / ratio_tag
        results_dir.mkdir(parents=True, exist_ok=True)

        for model_name in self.models:
            out_file = results_dir / f"{model_name}_{augmentation_name}.json"
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
            experiment_meta["reduction_method"] = self.reduction_method

            lineage = stage_cfg_eval.setdefault("lineage", {})
            lineage.setdefault("reduction_method", self.reduction_method)
            lineage["augmentation_name"] = (
                augmentation_name if augmentation_name != "baseline" else None
            )
            lineage["active_source"] = (
                "augmented" if augmentation_name != "baseline" else "reduced"
            )

            if augmentation_name == "baseline":
                hybea_path = lineage.get("reduced_hybea_path")
                base_root = lineage.get("reduced_base")
            else:
                hybea_path = lineage.get("augmented_hybea_paths", {}).get(augmentation_name)
                base_root = lineage.get("augmented_base")

            lineage["hybea_dataset_path"] = hybea_path
            lineage["hybea_dataset_base"] = base_root

            model_cls = MODEL_REGISTRY.get(model_name)
            logger.info("[STEP] → Evaluating model '%s' (augmentation=%s)", model_name, augmentation_name)
            model = model_cls(stage_cfg_eval)
            results = model.evaluate(dataset_reduced, dataset_augmented)
            logger.info("[SUCCESS] Model '%s' evaluation finished", model_name)

            wrote_results = False
            for plan in writer_plans:
                if plan.write_results:
                    with out_file.open("w", encoding="utf-8") as handle:
                        json.dump(results, handle, indent=2)
                    logger.info("💾 [%s] Saved results → %s", plan.name, out_file)
                    wrote_results = True

            if not wrote_results:
                with out_file.open("w", encoding="utf-8") as handle:
                    json.dump(results, handle, indent=2)
                logger.info("💾 Saved results → %s", out_file)
