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


def _extract_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Extract detailed statistics from a dataset.

    Returns:
        Dictionary containing:
        - kg1: Statistics for source KG (triples, entities, relations, attributes)
        - kg2: Statistics for target KG (triples, entities, relations, attributes)
        - aligned_pairs: Number of aligned entity pairs
    """
    from rdflib import Literal

    kg1 = dataset.knowledge_graph_source
    kg2 = dataset.knowledge_graph_target

    # Extract entities (subjects and objects that are URIRefs, not Literals)
    kg1_entities = set()
    kg2_entities = set()
    for s, p, o in kg1:
        kg1_entities.add(str(s))
        if not isinstance(o, Literal):
            kg1_entities.add(str(o))
    for s, p, o in kg2:
        kg2_entities.add(str(s))
        if not isinstance(o, Literal):
            kg2_entities.add(str(o))

    # Extract relations (predicates where object is URIRef) and attributes (predicates where object is Literal)
    kg1_relations = set()
    kg1_attributes = set()
    for s, p, o in kg1:
        if isinstance(o, Literal):
            kg1_attributes.add(str(p))
        else:
            kg1_relations.add(str(p))

    kg2_relations = set()
    kg2_attributes = set()
    for s, p, o in kg2:
        if isinstance(o, Literal):
            kg2_attributes.add(str(p))
        else:
            kg2_relations.add(str(p))

    return {
        "kg1": {
            "triples": len(kg1),
            "entities": len(kg1_entities),
            "relations": len(kg1_relations),
            "attributes": len(kg1_attributes),
        },
        "kg2": {
            "triples": len(kg2),
            "entities": len(kg2_entities),
            "relations": len(kg2_relations),
            "attributes": len(kg2_attributes),
        },
        "aligned_pairs": len(dataset.aligned_entities),
    }


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
        skip_dataset_write: bool = False,
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

        # Check if we can resume from cache
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
            if plan.write_reduced and not skip_dataset_write:
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

        # Extract detailed dataset statistics
        stats = _extract_dataset_statistics(dataset_reduced)

        StageSummaryWriter.write(
            reduction_root / "summary.json",
            {
                "method": self.reducer_name,
                "ratio": ratio,
                "target_entities": stage_cfg.get("reduction", {}).get("target_entities"),
                "aligned_pairs": stats["aligned_pairs"],
                "writer": writer_name,
                "statistics": stats,
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
        skip_dataset_write: bool = False,
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
        # Add stage_root to config so augmenter can save artifacts
        augmenter_config = dict(stage_cfg)
        augmenter_config.setdefault("augmentation", {})["stage_root"] = str(stage_root)
        augmenter = augmenter_cls(augmenter_config)
        dataset_augmented = augmenter.augment(dataset_reduced.clone())
        logger.info(
            "[SUCCESS] Augmentation '%s' complete (%d aligned pairs)",
            augmentation_name,
            len(dataset_augmented.aligned_entities),
        )

        # Track model directory for cleanup if needed (e.g., BART fine-tuned model)
        model_dir = stage_root / "model"
        if model_dir.exists():
            augmentation_paths["model"] = str(model_dir.resolve())

        for plan in self.writer_plans:
            plan_root = self._plan_root(stage_root, plan)
            if plan.write_augmented and not skip_dataset_write:
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

        # Extract detailed dataset statistics
        stats = _extract_dataset_statistics(dataset_augmented)

        StageSummaryWriter.write(
            stage_root / "summary.json",
            {
                "augmentation": augmentation_name,
                "ratio": ratio,
                "aligned_pairs": stats["aligned_pairs"],
                "writer": writer_name,
                "statistics": stats,
            },
        )

        # Validate that dataset is not None before returning
        if dataset_augmented is None:
            raise ValueError(
                f"Augmentation stage returned None dataset for '{augmentation_name}'. "
                f"This should never happen. Check augmenter implementation."
            )

        return dataset_augmented


class FilteringStage:
    """Encapsulate filtering logic for training mode selection.

    This stage filters datasets based on training mode:
    - "baseline": Returns original dataset (no synthetic data)
    - "synthetic_only": Returns only synthetic pairs (removes originals)
    - "augmented": Returns all pairs (original + synthetic)
    """

    def __init__(self, training_mode: str) -> None:
        """Initialize the filtering stage.

        Args:
            training_mode: One of "baseline", "synthetic_only", or "augmented"
        """
        self.training_mode = training_mode

        if training_mode not in ("baseline", "synthetic_only", "augmented"):
            logger.warning(
                f"Invalid training_mode '{training_mode}'. Defaulting to 'augmented'."
            )
            self.training_mode = "augmented"

    def execute(
        self,
        dataset_original: Dataset,
        dataset_augmented: Dataset,
    ) -> Dataset:
        """Filter dataset based on training mode.

        Args:
            dataset_original: Original dataset before augmentation
            dataset_augmented: Augmented dataset with original + synthetic pairs

        Returns:
            Filtered dataset based on training mode
        """
        if self.training_mode == "baseline":
            # Baseline mode: use only original pairs (no augmentation)
            logger.info(
                f"[FilteringStage] Using 'baseline' mode: {len(dataset_original.aligned_entities)} original pairs"
            )
            return dataset_original

        elif self.training_mode == "augmented":
            # Augmented mode: use all pairs (original + synthetic)
            logger.info(
                f"[FilteringStage] Using 'augmented' mode: {len(dataset_augmented.aligned_entities)} total pairs"
            )
            return dataset_augmented

        elif self.training_mode == "synthetic_only":
            # Synthetic-only mode: use ONLY synthetic pairs (remove originals)
            original_pairs = set(dataset_original.aligned_entities)
            augmented_pairs = set(dataset_augmented.aligned_entities)
            synthetic_pairs = augmented_pairs - original_pairs

            logger.info(f"[FilteringStage] Using 'synthetic_only' mode:")
            logger.info(f"  Original pairs: {len(original_pairs)}")
            logger.info(f"  Total augmented pairs: {len(augmented_pairs)}")
            logger.info(f"  Synthetic-only pairs: {len(synthetic_pairs)}")

            if not synthetic_pairs:
                logger.warning(
                    "[FilteringStage] No synthetic pairs found! Augmentation may have failed. "
                    "Returning augmented dataset as fallback."
                )
                return dataset_augmented

            # Create filtered dataset with only synthetic pairs
            dataset_filtered = dataset_augmented.clone()
            dataset_filtered.aligned_entities = tuple(sorted(synthetic_pairs))

            logger.info(
                f"[FilteringStage] Final training set: {len(dataset_filtered.aligned_entities)} synthetic pairs"
            )
            return dataset_filtered

        else:
            # Should never happen due to validation in __init__
            logger.error(
                f"[FilteringStage] Unknown training mode '{self.training_mode}'. Falling back to 'augmented'."
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

        # Use a single results.json file instead of per-model files
        out_file = evaluation_dir / "results.json"

        # Collect all model results
        all_results = {}
        for model_name in self.models:
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

            # Validate datasets before evaluation
            if dataset_reduced is None:
                raise ValueError(
                    f"dataset_reduced is None in evaluation for model '{model_name}'. "
                    f"This indicates a problem in the reduction stage."
                )
            if dataset_augmented is None:
                raise ValueError(
                    f"dataset_augmented is None in evaluation for model '{model_name}' "
                    f"(augmentation='{augmentation_name}'). This indicates a problem in the augmentation stage."
                )

            model_cls = get_alignment_model(model_name)
            logger.info("[STEP] → Evaluating model '%s' (augmentation=%s)", model_name, augmentation_name)
            model = model_cls(stage_cfg_eval)
            results = model.evaluate(dataset_reduced, dataset_augmented)
            logger.info("[SUCCESS] Model '%s' evaluation finished", model_name)

            all_results[model_name] = results

        # Write all results to a single file
        if all_results:
            self._write_results(out_file, all_results)
            evaluation_paths["results"] = str(out_file.resolve())

        # Merge evaluation metadata with existing summary if present
        summary_path = evaluation_dir / "summary.json"
        summary_data = {}

        # Read existing summary if it exists (from augmentation stage)
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary_data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                pass

        # Add evaluation metadata
        summary_data.update({
            "variant": variant_key,
            "models": sorted(self.models),
            "evaluation_files": sorted(evaluation_paths.values()),
        })

        StageSummaryWriter.write(summary_path, summary_data)

    @staticmethod
    def _resolve_evaluation_dir(
        lineage_cfg: Dict[str, Any],
        ratio_root: Path,
        augmentation_name: str,
    ) -> Path:
        # Save evaluation results directly in reduction/ or augmentation/
        # No subdirectories - files go directly in the stage directory
        if _is_baseline_variant(augmentation_name):
            target = ratio_root / "reduction"
        else:
            target = ratio_root / "augmentation"

        _ensure_directory(target)
        key = EvaluationStage._normalise_variant_key(augmentation_name)
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
