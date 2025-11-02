from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.alignment_models.methods.hybea import (
    runtime as hybea_runtime_state,
    apply_settings as hybea_apply_settings,
    path_for_KG as hybea_path_for_KG,
)
from src.alignment_models.methods.hybea.configuration import HybeaConfig
from src.alignment_models.methods.hybea.pipeline import HybeaPipeline
from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT, load_yaml
from src.core.dataset import Dataset
from src.core.dataset.writer import HybeaWriter
from src.logger import get_logger

logger = get_logger(__name__)


@MODEL_REGISTRY.register("hybea")
class HybEA:
    """Integration wrapper that executes the legacy HybEA attribute pipeline."""

    def __init__(self, config):
        self.stage_config = config or {}
        self.model_config = self._load_model_config()
        # logger.debug("[HybEA] Loaded configuration: %s", self.model_config)

    def evaluate(self, dataset_reduced: Dataset, dataset_augmented: Dataset | None) -> Dict[str, float]:
        dataset = dataset_augmented or dataset_reduced
        meta = self.stage_config.get("experiment", {})
        dataset_name = meta.get("dataset", "dataset")
        ratio = float(meta.get("ratio", self.model_config.reduction_ratio))

        logger.info(
            "[HybEA] Evaluating dataset '%s' (ratio=%.2f, aligned=%d)",
            dataset_name,
            ratio,
            len(dataset.aligned_entities),
        )
        logger.info("[STEP] HybEA evaluation starting")

        with tempfile.TemporaryDirectory(prefix="hybea_") as tmp_dir:
            workdir = Path(tmp_dir)
            export_root = workdir / dataset_name
            logger.info("[STEP] Exporting dataset to temporary workspace %s", export_root)
            self._export_dataset(dataset, export_root, dataset_name)

            ratio_tag = f"{ratio * 100:.1f}"
            iteration_dir = (
                workdir
                / "experiments"
                / ratio_tag
                / f"{dataset_name}_{self.model_config.structural_model}_{self.model_config.mode}"
            )
            iteration_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[STEP] Preparing HybEA iteration directory %s", iteration_dir)

            support_base_dir = self._resolve_support_base(dataset_name)
            logger.debug("[IMPORTANT] HybEA support directory resolved to %s", support_base_dir)

            hybea_apply_settings(
                self.model_config,
                dataset_name,
                base_dir=support_base_dir,
                workspace_dir=workdir,
                data_root=export_root,
                results_dir=iteration_dir,
                reduction_ratio=ratio,
            )

            self._prepare_support_artifacts(dataset_name, export_root)
            logger.info("[STEP] Support artefacts ready")

            pipeline = HybeaPipeline(
                dataset_name=dataset_name,
                ratio=ratio,
                iteration_dir=iteration_dir,
                mode=self.model_config.mode,
                structural_model=self.model_config.structural_model,
            )
            logger.info("[STEP] Launching HybEA pipeline")

            metrics = pipeline.run()
            logger.info("[SUCCESS] HybEA pipeline completed")
            logger.info(
                "[HybEA] Metrics: hits@1=%.4f hits@10=%.4f mrr=%.4f",
                metrics["hits@1"],
                metrics["hits@10"],
                metrics["mrr"],
            )
            return metrics

    def _export_dataset(self, dataset: Dataset, export_root: Path, dataset_name: str) -> None:
        writer = HybeaWriter()
        logger.debug("[HybEA] Exporting dataset to %s", export_root)
        writer.write(dataset, str(export_root), dataset_name=dataset_name)

    def _prepare_support_artifacts(self, dataset_name: str, export_root: Path) -> None:
        ratio_tag = str(round(hybea_runtime_state.SIZE_AFTER_REDUCTION_IN_PERCENTAGE, 1))
        base_dir = Path(hybea_runtime_state.BASE_DIR)
        names_dir = base_dir / "data" / "entity_names" / ratio_tag / dataset_name
        names_dir.mkdir(parents=True, exist_ok=True)

        kg1_file, kg2_file = hybea_path_for_KG(dataset_name)
        analysis_targets = [
            names_dir / kg1_file.replace("names", "analysis"),
            names_dir / kg2_file.replace("names", "analysis"),
        ]
        if any(not path.exists() for path in analysis_targets):
            try:
                from src.alignment_models.methods.hybea.src.generate_names.name_analysis import run_name_analysis

                logger.info("[HybEA] Generating name analysis workbooks for %s", dataset_name)
                run_name_analysis()
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("[HybEA] Failed to generate name analysis files: %s", exc)

        name_targets = [names_dir / kg1_file, names_dir / kg2_file]
        if any(not path.exists() for path in name_targets):
            try:
                from src.alignment_models.methods.hybea.src.generate_names.prioritize_names import run_prioritize

                logger.info("[HybEA] Generating prioritized names for %s", dataset_name)
                run_prioritize()
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("[HybEA] Failed to generate prioritized name files: %s", exc)

        # Fallback to quick TSV export if official generators did not create the Excel files.
        self._write_entity_names(export_root / "attribute_data" / "ent_ids_1", names_dir / kg1_file)
        self._write_entity_names(export_root / "attribute_data" / "ent_ids_2", names_dir / kg2_file)

        # Copy Excel files to the location where tools.py expects them
        self._copy_entity_names(names_dir / kg1_file, export_root / "attribute_data" / kg1_file)
        self._copy_entity_names(names_dir / kg2_file, export_root / "attribute_data" / kg2_file)

        if self.model_config.structural_model.lower() == "rrea":
            self._prepare_rrea_inputs(dataset_name, export_root)

    def _resolve_support_base(self, dataset_name: str) -> Path:
        """Return the base directory for HybEA support artefacts."""

        lineage = self.stage_config.get("lineage", {})
        source = lineage.get("active_source", "reduced")
        reduction_method = lineage.get("reduction_method")
        augmentation_name = lineage.get("augmentation_name")
        ratio_tag = lineage.get("ratio_tag") or self.stage_config.get("experiment", {}).get("ratio_tag")

        support_root = (PROJECT_ROOT / "data" / "hybea_support").resolve()
        components = [source]
        if reduction_method:
            components.append(reduction_method)
        if source == "augmented" and augmentation_name:
            components.append(augmentation_name)

        hybea_dataset_path = lineage.get("hybea_dataset_path")
        hybea_dataset_base = lineage.get("hybea_dataset_base")
        writer_components: list[str] = []

        if hybea_dataset_path and hybea_dataset_base:
            try:
                dataset_path = Path(hybea_dataset_path).resolve()
                base_root = Path(hybea_dataset_base).resolve()
                relative_parts = list(dataset_path.relative_to(base_root).parts)

                if ratio_tag and relative_parts and relative_parts[-1] == ratio_tag:
                    relative_parts.pop()
                if relative_parts and relative_parts[-1] == dataset_name:
                    relative_parts.pop()
                if source == "augmented" and augmentation_name and relative_parts and relative_parts[0] == augmentation_name:
                    relative_parts.pop(0)

                writer_components = relative_parts
            except Exception:
                writer_components = []

        if not writer_components:
            writer_components = ["hybea"]

        components.extend(writer_components)
        return support_root.joinpath(*components)

    def _write_entity_names(self, ent_ids_path: Path, target_xlsx: Path) -> None:
        if target_xlsx.exists():
            return
        if not ent_ids_path.exists():
            logger.warning("[HybEA] Missing ent_id file %s; skipping name export", ent_ids_path)
            return
        df = pd.read_csv(ent_ids_path, sep="\t", header=None, names=["e1", "uri"])
        df["name"] = df["uri"].map(self._friendly_name)
        df.to_excel(target_xlsx, index=False)

    def _copy_entity_names(self, source_xlsx: Path, target_xlsx: Path) -> None:
        """Copy entity names Excel file from source to target location."""
        if not source_xlsx.exists():
            logger.debug("[HybEA] Source entity names file %s does not exist; skipping copy", source_xlsx)
            return
        target_xlsx.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.copy2(source_xlsx, target_xlsx)
            logger.debug("[HybEA] Copied entity names from %s to %s", source_xlsx, target_xlsx)
        except Exception as exc:
            logger.warning("[HybEA] Failed to copy entity names file: %s", exc)

    @staticmethod
    def _friendly_name(uri: str) -> str:
        if not isinstance(uri, str):
            return "no_value"
        fragment = uri.split("/")[-1]
        fragment = fragment.split(":")[-1]
        fragment = fragment.replace("_", " ")
        fragment = fragment.strip()
        return fragment or "no_value"

    def _prepare_rrea_inputs(self, dataset_name: str, export_root: Path) -> None:
        attribute_dir = export_root / "attribute_data"
        rrea_dir = export_root / "rrea_data" / dataset_name
        rrea_dir.mkdir(parents=True, exist_ok=True)

        uri_to_id, id_to_uri = self._read_entity_maps(attribute_dir)
        rel_map_1 = self._read_relation_map(attribute_dir / "rel_ids_1", offset=0)
        rel_offset = (max(rel_map_1.values()) + 1) if rel_map_1 else 0
        rel_map_2 = self._read_relation_map(attribute_dir / "rel_ids_2", offset=rel_offset)

        self._convert_triples(attribute_dir / "triples_1", rrea_dir / "triples_1", uri_to_id, rel_map_1)
        self._convert_triples(attribute_dir / "triples_2", rrea_dir / "triples_2", uri_to_id, rel_map_2)
        self._convert_pairs(attribute_dir / "sup_pairs", rrea_dir / "sup_pairs")
        self._convert_pairs(attribute_dir / "ref_pairs", rrea_dir / "ref_pairs")
        combined_rel_map = {**rel_map_1, **rel_map_2}
        self._write_rrea_vocab(rrea_dir / "vocab.txt", id_to_uri, combined_rel_map)

    def _read_entity_maps(
        self,
        attribute_dir: Path,
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        uri_to_id: Dict[str, int] = {}
        id_to_uri: Dict[int, str] = {}

        for filename in ("ent_ids_1", "ent_ids_2"):
            path = attribute_dir / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    idx = int(parts[0])
                    uri = parts[1]
                    uri_to_id[uri] = idx
                    id_to_uri[idx] = uri
        return uri_to_id, id_to_uri

    def _read_relation_map(self, file_path: Path, *, offset: int) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        if not file_path.exists():
            return mapping
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                idx = int(parts[0]) + offset
                rel = parts[1]
                mapping[rel] = idx
        return mapping

    def _convert_triples(
        self,
        source: Path,
        destination: Path,
        entity_map: Dict[str, int],
        relation_map: Dict[str, int],
    ) -> None:
        if not source.exists() or not relation_map:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("r", encoding="utf-8") as reader, destination.open(
            "w", encoding="utf-8"
        ) as writer:
            for line in reader:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                subj_id = entity_map.get(parts[0])
                rel_id = relation_map.get(parts[1])
                obj_id = entity_map.get(parts[2])
                if None in (subj_id, rel_id, obj_id):
                    continue
                writer.write(f"{subj_id} {rel_id} {obj_id}\n")

    def _convert_pairs(self, source: Path, destination: Path) -> None:
        if not source.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("r", encoding="utf-8") as reader, destination.open(
            "w", encoding="utf-8"
        ) as writer:
            for line in reader:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                writer.write(f"{parts[0]} {parts[1]}\n")

    def _write_rrea_vocab(
        self,
        destination: Path,
        entity_ids: Dict[int, str],
        relation_map: Dict[str, int],
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for idx in sorted(entity_ids.keys()):
                handle.write(f"{idx}\t{entity_ids[idx]}\n")
            for rel_uri, idx in sorted(relation_map.items(), key=lambda item: item[1]):
                handle.write(f"{idx}\t{rel_uri}\n")

    def _load_model_config(self) -> HybeaConfig:
        path = PROJECT_ROOT / "config/models/hybea.yaml"
        payload = load_yaml(path)
        model_section = payload.get("model", {})

        overrides = (
            (PROJECT_ROOT / "config/models/hybea.local.yaml")
            if (PROJECT_ROOT / "config/models/hybea.local.yaml").exists()
            else None
        )
        if overrides:
            model_section = {
                **model_section,
                **load_yaml(overrides).get("model", {}),
            }

        stage_overrides = (
            (PROJECT_ROOT / "config/models/hybea.stage.yaml")
            if (PROJECT_ROOT / "config/models/hybea.stage.yaml").exists()
            else None
        )
        if stage_overrides:
            model_section = {
                **model_section,
                **load_yaml(stage_overrides).get("model", {}),
            }

        stage_override = self.stage_config.get("models", {}).get("hybea", {})
        if stage_override:
            model_section = {**model_section, **stage_override}

        return HybeaConfig.from_dict(model_section)

    @staticmethod
    def _compute_alignment_metrics(
        ent_pairs: Iterable[Tuple[int, int]],
        similarity_matrix,
    ) -> Dict[str, float]:
        sim = similarity_matrix.detach().cpu().numpy()
        pairs = list(ent_pairs)
        if not pairs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

        hits1 = 0
        hits10 = 0
        rr_total = 0.0

        for idx, (_, tgt) in enumerate(pairs):
            scores = sim[idx]
            ranking = np.argsort(-scores)
            try:
                rank = int(np.where(ranking == tgt)[0][0])
            except IndexError:
                # target not found; treat as worst-case rank
                rank = len(ranking)
            if rank == 0:
                hits1 += 1
            if rank < 10:
                hits10 += 1
            rr_total += 1.0 / (rank + 1)

        total = len(pairs)
        hits1_rate = hits1 / total
        hits10_rate = hits10 / total
        mrr = rr_total / total

        # Legacy precision/recall approximated by hits@1
        return {
            "precision": hits1_rate,
            "recall": hits1_rate,
            "f1": hits1_rate,
            "hits@1": hits1_rate,
            "hits@10": hits10_rate,
            "mrr": mrr,
        }


__all__ = ["HybEA"]
