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


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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

        if self.resume and self._has_cached(reader_root):
            logger.info(
                "⏭️  Skipping reduction — cached artefacts detected (ratio=%s)", ratio_tag
            )
            self._record_plan_paths(
                reduction_root,
                reduced_paths,
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
            self._maybe_record_plan(plan, plan_root, reduced_paths, reduction_paths, reduction_meta, lineage)

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
        reduction_paths: Dict[str, str],
        reduction_meta: Dict[str, Any],
        lineage: Dict[str, Any],
    ) -> None:
        for plan in self.writer_plans:
            plan_root = base_root / plan.name
            if plan_root.exists():
                reduced_paths[plan.name] = str(plan_root)
                reduction_paths[plan.name] = str(plan_root.resolve())
                if getattr(plan.writer, "file_type", None) == "hybea" or plan.name == "hybea":
                    lineage["reduced_hybea_path"] = str(plan_root)
                    reduction_meta["hybea_path"] = str(plan_root.resolve())

    @staticmethod
    def _maybe_record_plan(plan, plan_root, reduced_paths, reduction_paths, reduction_meta, lineage):
        if plan_root.exists():
            reduced_paths[plan.name] = str(plan_root)
            reduction_paths[plan.name] = str(plan_root.resolve())
            if getattr(plan.writer, "file_type", None) == "hybea" or plan.name == "hybea":
                lineage["reduced_hybea_path"] = str(plan_root)
                reduction_meta["hybea_path"] = str(plan_root.resolve())


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
                    per_aug_paths[plan.name] = str(plan_root)
                    augmentation_paths[plan.name] = str(plan_root.resolve())
                    if getattr(plan.writer, "file_type", None) == "hybea" or plan.name == "hybea":
                        hybea_aug_paths[augmentation_name] = str(plan_root)
                        augmentation_meta["hybea_path"] = str(plan_root.resolve())
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
                per_aug_paths[plan.name] = str(plan_root)
                augmentation_paths[plan.name] = str(plan_root.resolve())
                if getattr(plan.writer, "file_type", None) == "hybea" or plan.name == "hybea":
                    hybea_aug_paths[augmentation_name] = str(plan_root)
                    augmentation_meta["hybea_path"] = str(plan_root.resolve())

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
                augmentation_name if augmentation_name not in (None, "baseline") else None
            )
            lineage["active_source"] = (
                "augmented" if augmentation_name not in (None, "baseline") else "reduced"
            )

            if augmentation_name in (None, "baseline"):
                hybea_path = lineage.get("reduced_hybea_path")
                base_root = lineage_cfg.get("reduced_base")
            else:
                hybea_path = lineage.get("augmented_hybea_paths", {}).get(augmentation_name)
                base_root = lineage_cfg.get("augmented_base")

            lineage["hybea_dataset_path"] = hybea_path
            lineage["hybea_dataset_base"] = base_root

            model_cls = get_alignment_model(model_name)
            logger.info("[STEP] → Evaluating model '%s' (augmentation=%s)", model_name, augmentation_name)
            model = model_cls(stage_cfg_eval)
            results = model.evaluate(dataset_reduced, dataset_augmented)
            logger.info("[SUCCESS] Model '%s' evaluation finished", model_name)

            self._write_results(out_file, results)

        StageSummaryWriter.write(
            evaluation_dir / "summary.json",
            {
                "augmentation": augmentation_name if augmentation_name not in (None, "baseline") else None,
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
        if not name or name == "baseline":
            return "reduced"
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
