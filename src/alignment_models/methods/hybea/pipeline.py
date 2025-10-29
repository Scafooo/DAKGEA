"""Orchestrator for the hybrid HybEA pipeline.

The original project implemented its experiment loop inside `hybea.py`,
coupling file-system side effects with alternating attribute/structure
refinements.  This module reproduces the high-level control flow while
keeping state in memory, so it can be driven by the DAKGEA experiment
runner without mutating the repository tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from src.alignment_models.methods.hybea import runtime as cfg
from src.alignment_models.methods.hybea.src.attribute_model.attr_main import run_attr_model
from src.logger import get_logger

logger = get_logger(__name__)


def _reciprocity(
    ents_1: Sequence[int],
    ents_2: Sequence[int],
    res_mat_1: np.ndarray,
    res_mat_2: np.ndarray,
) -> Set[Tuple[int, int]]:
    max_indices_1 = np.argmax(res_mat_1, axis=1)
    max_indices_2 = np.argmax(res_mat_2, axis=1)
    pairs: Set[Tuple[int, int]] = set()
    for i in range(len(ents_1)):
        if max_indices_1[i] == max_indices_2[i]:
            pairs.add((ents_1[i], ents_2[i]))
    return pairs


def _to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


@dataclass
class StageOutcome:
    aligned_pairs: List[Tuple[int, int]]
    similarity: np.ndarray
    metrics: Dict[str, float]


class HybeaPipeline:
    """Execute the alternating HybEA pipeline inside a temporary workspace."""

    def __init__(
        self,
        dataset_name: str,
        ratio: float,
        iteration_dir: Path,
        mode: str,
        structural_model: str,
    ) -> None:
        self.dataset = dataset_name
        self.ratio = ratio
        self.iteration_dir = iteration_dir
        self.mode = mode
        self.structural_model = structural_model.lower()

        self.uri_to_id: Dict[str, int] = {}
        self.id_to_uri: Dict[int, str] = {}
        self.kg1_ids: Set[int] = set()
        self.kg2_ids: Set[int] = set()

        self.attr_history: List[Set[Tuple[str, str]]] = []
        self.struct_history: List[Set[Tuple[str, str]]] = []
        self.added_entities: Set[str] = set()
        self.latest_outcome: Optional[StageOutcome] = None

        self._load_entity_mappings()

        (self.turn, self.stop_attribute, self.stop_structure, self.max_turn) = self._initial_state()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, float]:
        while True:
            if self.max_turn is not None and self.turn >= self.max_turn:
                logger.debug("[HybEA] Reached turn limit (%d)", self.max_turn)
                break

            attr_enabled = not self.stop_attribute
            struct_enabled = not self.stop_structure
            logger.info(
                "[HybEA] Turn %d | mode=%s | attribute_enabled=%s | structural_enabled=%s",
                self.turn,
                self.mode,
                attr_enabled,
                struct_enabled,
            )

            if self.turn % 2 == 0:  # attribute stage
                new_pairs = self._run_attribute_stage()
                self.turn += 1
                if self.stop_attribute:
                    logger.info("[HybEA] Attribute stage flagged as final turn; stopping pipeline")
                    break
            else:  # structure stage
                if self.stop_structure:
                    logger.info(
                        "[HybEA] Structural stage disabled (mode=%s, structure.enabled=%s); finishing after attribute stage",
                        self.mode,
                        struct_enabled,
                    )
                    break
                new_pairs = self._run_structure_stage()
                self.turn += 1

            if not new_pairs:
                logger.debug("[HybEA] No new pairs discovered; terminating pipeline")
                break

        if self.latest_outcome is None:
            logger.warning("[HybEA] Pipeline finished without any evaluation outcome")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

        return self.latest_outcome.metrics

    # ------------------------------------------------------------------
    # stage helpers
    # ------------------------------------------------------------------

    def _run_attribute_stage(self) -> Set[Tuple[str, str]]:
        logger.debug("[HybEA] Attribute stage starting (history attr=%d struct=%d)", len(self.attr_history), len(self.struct_history))
        new_pairs_indices = self._collect_all_pair_indices()

        ent_ill, res_mat_1, res_mat_2, _, _ = run_attr_model(new_pairs_indices)
        left_ids = [self._tensor_index(pair[0]) for pair in ent_ill]
        right_ids = [self._tensor_index(pair[1]) for pair in ent_ill]

        res_mat_1 = _to_numpy(res_mat_1)
        res_mat_2 = _to_numpy(res_mat_2)
        results = _reciprocity(left_ids, right_ids, res_mat_1, res_mat_2)

        new_pairs: Set[Tuple[str, str]] = set()
        for left_idx, right_idx in results:
            left_uri = self.id_to_uri.get(left_idx)
            right_uri = self.id_to_uri.get(right_idx)
            if left_uri is None or right_uri is None or left_uri in self.added_entities:
                continue
            self.added_entities.add(left_uri)
            new_pairs.add((left_uri, right_uri))

        self.attr_history.append(new_pairs)
        self._update_metrics(left_ids, right_ids, res_mat_1)
        logger.info("[HybEA] Attribute stage found %d reciprocal pairs", len(new_pairs))
        return new_pairs

    def _run_structure_stage(self) -> Set[Tuple[str, str]]:
        logger.debug("[HybEA] Structural stage starting (history attr=%d struct=%d)", len(self.attr_history), len(self.struct_history))

        new_pairs_uri = self._gather_pair_history()
        new_pairs_indices = self._collect_all_pair_indices()

        if self.structural_model == "knowformer":
            from src.alignment_models.methods.hybea.src.structure_model.structure_main import run_structure_model

            res_mat_1, res_mat_2, ents_1, ents_2, vocab, _, _ = run_structure_model(new_pairs_uri)

            vocab_inv = {v: k for k, v in vocab.items()}
            left_ids: List[int] = []
            right_ids: List[int] = []
            for entry in ents_1:
                token = vocab_inv.get(self._tensor_index(entry))
                left_ids.append(self.uri_to_id.get(token, -1))
            for entry in ents_2:
                token = vocab_inv.get(self._tensor_index(entry))
                right_ids.append(self.uri_to_id.get(token, -1))

            res_mat_1 = _to_numpy(res_mat_1)
            res_mat_2 = _to_numpy(res_mat_2)
            results = _reciprocity(left_ids, right_ids, res_mat_1, res_mat_2)

            new_pairs: Set[Tuple[str, str]] = set()
            for left_idx, right_idx in results:
                left_uri = self.id_to_uri.get(left_idx)
                right_uri = self.id_to_uri.get(right_idx)
                if left_uri is None or right_uri is None or left_uri in self.added_entities:
                    continue
                self.added_entities.add(left_uri)
                new_pairs.add((left_uri, right_uri))

            self.struct_history.append(new_pairs)
            self._update_metrics(left_ids, right_ids, res_mat_1)
            logger.info("[HybEA] Structural stage (Knowformer) found %d reciprocal pairs", len(new_pairs))
            return new_pairs

        if self.structural_model == "rrea":
            from src.alignment_models.methods.hybea.src.structure_model.rrea.rrea_main import run_rrea_structural

            res_mat_1, res_mat_2, indices_L, indices_R = run_rrea_structural(list(new_pairs_indices))
            res_mat_1 = _to_numpy(res_mat_1)
            res_mat_2 = _to_numpy(res_mat_2)
            left_ids = [int(idx) for idx in indices_L]
            right_ids = [int(idx) for idx in indices_R]

            results = _reciprocity(left_ids, right_ids, res_mat_1, res_mat_2)
            new_pairs: Set[Tuple[str, str]] = set()
            for left_idx, right_idx in results:
                left_uri = self.id_to_uri.get(left_idx)
                right_uri = self.id_to_uri.get(right_idx)
                if left_uri is None or right_uri is None or left_uri in self.added_entities:
                    continue
                self.added_entities.add(left_uri)
                new_pairs.add((left_uri, right_uri))

            self.struct_history.append(new_pairs)
            self._update_metrics(left_ids, right_ids, res_mat_1)
            logger.info("[HybEA] Structural stage (RREA) found %d reciprocal pairs", len(new_pairs))
            return new_pairs

        logger.warning("[HybEA] Unknown structural model '%s'; skipping stage", self.structural_model)
        return set()

    # ------------------------------------------------------------------
    # utility helpers
    # ------------------------------------------------------------------

    def _load_entity_mappings(self) -> None:
        base_path = Path(cfg.DATA_PATH)
        for filename, target_set in (("ent_ids_1", self.kg1_ids), ("ent_ids_2", self.kg2_ids)):
            file_path = base_path / filename
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    idx = int(parts[0])
                    uri = parts[1]
                    self.id_to_uri[idx] = uri
                    self.uri_to_id[uri] = idx
                    target_set.add(idx)

    def _initial_state(self) -> Tuple[int, bool, bool, Optional[int]]:
        mode = self.mode.lower()
        stop_attr = False
        stop_struct = False
        max_turn: Optional[int] = None

        if mode == "hybea":
            turn = 2
        elif mode == "hybea_struct_first":
            turn = 1
        elif mode == "hybea_without_structure":
            turn = 2
            stop_struct = True
        elif mode == "hybea_without_factual":
            turn = 1
            stop_attr = True
        elif mode == "hybea_basic":
            turn = 2
            max_turn = 4  # exits before executing turn 4 (attr -> struct only)
        elif mode == "hybea_basic_structure_first":
            turn = 1
            max_turn = 3  # structure -> attribute
        else:
            logger.warning("[HybEA] Unknown mode '%s'; defaulting to attribute-only", self.mode)
            turn = 2
            stop_struct = True

        structure_enabled = getattr(cfg, "DO_TRAIN", True)
        if not structure_enabled:
            stop_struct = True

        if structure_enabled and stop_attr:
            logger.info("[HybEA] Structure-only mode active")

        return turn, stop_attr, stop_struct, max_turn

    def _collect_pair_indices(
        self,
        history: Iterable[Set[Tuple[str, str]]],
    ) -> Set[Tuple[int, int]]:
        collected: Set[Tuple[int, int]] = set()
        for pairs in history:
            for left_uri, right_uri in pairs:
                left_idx = self.uri_to_id.get(left_uri)
                right_idx = self.uri_to_id.get(right_uri)
                if left_idx is None or right_idx is None:
                    continue
                collected.add((left_idx, right_idx))
        return collected

    def _collect_all_pair_indices(self) -> Set[Tuple[int, int]]:
        combined = self._collect_pair_indices(self.attr_history)
        combined.update(self._collect_pair_indices(self.struct_history))
        return combined

    def _update_metrics(
        self,
        ent_pairs_left: Sequence[int],
        ent_pairs_right: Sequence[int],
        similarity: np.ndarray,
    ) -> None:
        aligned_pairs = list(zip(ent_pairs_left, ent_pairs_right))
        metrics = self._compute_alignment_metrics(aligned_pairs, similarity)
        self.latest_outcome = StageOutcome(aligned_pairs, similarity, metrics)

    @staticmethod
    def _tensor_index(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if hasattr(value, "__int__"):
            return int(value)
        return int(value)

    def _gather_pair_history(self) -> Set[Tuple[str, str]]:
        combined: Set[Tuple[str, str]] = set()
        for history in self.attr_history:
            combined.update(history)
        for history in self.struct_history:
            combined.update(history)
        return combined

    @staticmethod
    def _compute_alignment_metrics(
        ent_pairs: Iterable[Tuple[int, int]],
        similarity: np.ndarray,
    ) -> Dict[str, float]:
        ent_pairs = list(ent_pairs)
        if not ent_pairs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

        hits1 = 0
        hits10 = 0
        rr_total = 0.0

        for idx, (_, target) in enumerate(ent_pairs):
            if idx >= similarity.shape[0]:
                break
            scores = similarity[idx]
            ranking = np.argsort(-scores)
            try:
                rank = int(np.where(ranking == target)[0][0])
            except IndexError:
                rank = len(ranking)
            if rank == 0:
                hits1 += 1
            if rank < 10:
                hits10 += 1
            rr_total += 1.0 / (rank + 1)

        total = len(ent_pairs)
        hits1_rate = hits1 / total if total else 0.0
        hits10_rate = hits10 / total if total else 0.0
        mrr = rr_total / total if total else 0.0

        return {
            "precision": hits1_rate,
            "recall": hits1_rate,
            "f1": hits1_rate,
            "hits@1": hits1_rate,
            "hits@10": hits10_rate,
            "mrr": mrr,
        }
