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


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


class _WriterStage:
    """Shared helpers for stages that materialize datasets to disk."""

    def __init__(
        self,
        stage_key: str,
        writer_plans: Iterable[WriterPlan],
        resume: bool,
    ) -> None:
        self.stage_key = stage_key
        self.writer_plans = list(writer_plans)
        self.resume = resume

    def _ensure_reader_plan(
        self,
        lineage: Dict[str, Any],
        reader_hint: Optional[Any] = None,
    ) -> str:
        bucket = lineage.setdefault("reader_plan", {})
        plan_name = bucket.get(self.stage_key)
        if plan_name is None:
            plan_name = self._select_reader_plan(reader_hint)
            bucket[self.stage_key] = plan_name
        return plan_name

    def _select_reader_plan(self, reader_hint: Optional[Any] = None) -> str:
        if self.writer_plans:
            return self._plan_name(self.writer_plans[0].name)
        if reader_hint is not None:
            format_name = getattr(reader_hint, "file_type", None) or "default"
            return self._plan_name(format_name)
        return self._plan_name("default")

    @staticmethod
    def _has_cached(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    @staticmethod
    def _read_cached_dataset(reader, reader_root: Path, subtype: Optional[str]):
        try:
            return reader.read(str(reader_root), subtype=subtype)
        except FileNotFoundError:
            return None

    def _record_plan_paths(
        self,
        base_root: Path,
        path_bucket: Dict[str, str],
        lineage: Optional[Dict[str, Any]] = None,
    ) -> None:
        for plan in self.writer_plans:
            plan_root = self._plan_root(base_root, plan)
            if plan_root.exists():
                path_bucket[plan.name] = str(plan_root.resolve())
                if lineage is not None:
                    lineage["dataset_workspace"] = str(plan_root.resolve())

    @staticmethod
    def _plan_root(base_root: Path, plan: WriterPlan) -> Path:
        return base_root / "dataset" / plan.name

    @staticmethod
    def _plan_name(plan_name: str) -> str:
        return f"dataset/{plan_name}"


class ReductionStage(_WriterStage):
    """Encapsulate the reduction logic and caching behaviour."""

    def __init__(
        self,
        reducer_name: str,
        writer_plans: Iterable[WriterPlan],
        resume: bool,
    ) -> None:
        super().__init__("reduction", writer_plans, resume)
        self.reducer_cls = REDUCTION_REGISTRY.get(reducer_name)
        self.reducer_name = reducer_name

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
        reader_plan = self._ensure_reader_plan(lineage)
        reader_root = reduction_root / reader_plan

        reduction_meta = ratio_meta.setdefault(
            "reduction", {"method": self.reducer_name, "paths": {}}
        )
        reduction_meta["method"] = self.reducer_name
        reduction_meta["reader_plan"] = reader_plan
        reduction_paths = reduction_meta.setdefault("paths", {})
        reduction_meta["summary"] = str((reduction_root / "summary.json").resolve())

        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping reduction — cached artefacts detected (ratio=%s)", ratio_tag
            )
            self._record_plan_paths(reduction_root, reduction_paths, lineage)
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
            plan_root = self._plan_root(reduction_root, plan)
            if plan.write_reduced:
                _ensure_directory(plan_root)
                plan.writer.write(dataset_reduced, str(plan_root))
                logger.info("📝 [%s] Saved reduced dataset → %s", plan.name, plan_root)
                reduction_paths[plan.name] = str(plan_root.resolve())
                lineage["dataset_workspace"] = str(plan_root.resolve())
            elif plan_root.exists():
                reduction_paths[plan.name] = str(plan_root.resolve())
                lineage["dataset_workspace"] = str(plan_root.resolve())

        # Get the writer name (single writer per stage)
        writer_name = None
        for plan in self.writer_plans:
            if plan.write_reduced:
                writer_name = plan.name
                break

        StageSummaryWriter.write(
            reduction_root / "summary.json",
            {
                "method": self.reducer_name,
                "ratio": ratio,
                "target_entities": stage_cfg.get("reduction", {}).get("target_entities"),
                "aligned_pairs": len(dataset_reduced.aligned_entities),
                "writer": writer_name,
            },
        )
        return dataset_reduced


class AugmentationStage(_WriterStage):
    """Encapsulate augmentation logic and artefact management."""

    def __init__(
        self,
        writer_plans: Iterable[WriterPlan],
        resume: bool,
    ) -> None:
        super().__init__("augmentation", writer_plans, resume)

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
        stage_root = augmentation_root
        _ensure_directory(stage_root)
        reader_plan = self._ensure_reader_plan(lineage, reader)
        reader_root = stage_root / reader_plan

        augmentations_meta = ratio_meta.setdefault("augmentations", {})
        augmentation_meta = augmentations_meta.setdefault(
            augmentation_name, {"paths": {}, "reader_plan": reader_plan}
        )
        augmentation_meta["reader_plan"] = reader_plan
        augmentation_meta["summary"] = str((stage_root / "summary.json").resolve())
        augmentation_paths = augmentation_meta.setdefault("paths", {})

        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping augmentation '%s' — cached artefacts detected (ratio=%s)",
                augmentation_name,
                ratio_tag,
            )
            for plan in self.writer_plans:
                plan_root = self._plan_root(stage_root, plan)
                if plan_root.exists():
                    augmentation_paths[plan.name] = str(plan_root.resolve())
            cached = self._read_cached_dataset(reader, reader_root, subtype)
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
            plan_root = self._plan_root(stage_root, plan)
            if plan.write_augmented:
                _ensure_directory(plan_root)
                plan.writer.write(dataset_augmented, str(plan_root))
                logger.info("📝 [%s] Saved augmented dataset → %s", plan.name, plan_root)
                augmentation_paths[plan.name] = str(plan_root.resolve())
            elif plan_root.exists():
                augmentation_paths[plan.name] = str(plan_root.resolve())

        # Get the writer name (single writer per stage)
        writer_name = None
        for plan in self.writer_plans:
            if plan.write_augmented:
                writer_name = plan.name
                break

        StageSummaryWriter.write(
            stage_root / "summary.json",
            {
                "augmentation": augmentation_name,
                "ratio": ratio,
                "aligned_pairs": len(dataset_augmented.aligned_entities),
                "writer": writer_name,
            },
        )
        return dataset_augmented


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

            # Add lineage information for model's reference
            lineage = stage_cfg_eval.setdefault("lineage", {})
            lineage["augmentation_name"] = (
                augmentation_name if not _is_baseline_variant(augmentation_name) else None
            )
            lineage["variant"] = variant_key

            # Copy essential paths from lineage_cfg that models might need
            for key in ["evaluation_root", "ratio_root", "raw_source"]:
                if key in lineage_cfg:
                    lineage[key] = lineage_cfg[key]

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
