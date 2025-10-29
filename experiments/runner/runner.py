"""Core experiment orchestration logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.alignment_models.registry import MODEL_REGISTRY
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.config.loader import Config
from src.dataset.Dataset import Dataset
from src.dataset.reader.ReaderFactory import ReaderFactory
from src.dataset.writer.WriterFactory import WriterFactory
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

                for ratio in self.ratios:
                    ratio_desc = f"{ratio * 100:.1f}%"
                    progress.set_description(f"📦 {spec.name} [{ratio_desc}]")
                    stage_cfg = self._build_stage_config(
                        spec.name, len(dataset.aligned_entities), ratio
                    )
                    dataset_reduced = self._perform_reduction(
                        reader, writer_plans, dataset, spec, ratio, stage_cfg
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
            }
        )
        logger.debug(
            "Stage config prepared for dataset=%s ratio=%.3f (target_entities=%d)",
            dataset_name,
            ratio,
            reduction_cfg["target_entities"],
        )
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

        reducer = self.reducer_cls(stage_cfg)
        dataset_reduced = reducer.reduce(dataset)

        for plan in writer_plans:
            if plan.write_reduced:
                plan_root = (
                    self.base_reduced
                    / self.reduction_method
                    / plan.name
                    / spec.name
                    / ratio_tag
                )
                plan.writer.write(dataset_reduced, str(plan_root))
                logger.info("📝 [%s] Saved reduced dataset → %s", plan.name, plan_root)

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
        augmenter = augmenter_cls(stage_cfg)
        dataset_augmented = augmenter.augment(dataset_reduced)

        for plan in writer_plans:
            if plan.write_augmented:
                plan_root = (
                    self.base_augmented
                    / self.reduction_method
                    / augmentation_name
                    / plan.name
                    / spec.name
                    / ratio_tag
                )
                plan.writer.write(dataset_augmented, str(plan_root))
                logger.info("📝 [%s] Saved augmented dataset → %s", plan.name, plan_root)

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

            model_cls = MODEL_REGISTRY.get(model_name)
            model = model_cls(stage_cfg)
            results = model.evaluate(dataset_reduced, dataset_augmented)

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
