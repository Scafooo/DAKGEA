"""BERT-INT integration: configuration, training orchestration, and evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .basic_training import BasicUnitArtifacts, train_basic_unit_model
from .basic_unit_model import BasicBertUnitModel
from .config import BertIntConfig, load_bert_int_config
from .data import PreparedBertIntData, prepare_basic_unit_data
from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_INPUT_SIZE = 768
DEFAULT_RESULT_SIZE = 768


def _batch_entities(entity_ids: Sequence[str], batch_size: int):
    step = max(1, batch_size)
    for idx in range(0, len(entity_ids), step):
        yield entity_ids[idx : idx + step]


def _compute_embeddings(
    model: BasicBertUnitModel,
    entity_ids: Sequence[str],
    entity_inputs: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for batch in _batch_entities(entity_ids, batch_size):
            if not batch:
                continue
            token_ids = torch.stack([entity_inputs[e]["input_ids"] for e in batch], dim=0).to(device).long()
            attention_mask = torch.stack([entity_inputs[e]["attention_mask"] for e in batch], dim=0).to(device).long()
            batch_emb = model(token_ids, attention_mask)
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
            for entity_id, embedding in zip(batch, batch_emb.detach().cpu()):
                embeddings[entity_id] = embedding
    return embeddings


def _compute_directional_ranks(
    pairs: Sequence[Tuple[str, str]],
    source_embeddings: Dict[str, torch.Tensor],
    target_embeddings: Dict[str, torch.Tensor],
    ordered_targets: Sequence[str],
) -> List[int]:
    if not pairs or not ordered_targets:
        return []

    target_list = [entity for entity in ordered_targets if entity in target_embeddings]
    if not target_list:
        return []

    target_matrix = torch.stack([target_embeddings[e] for e in target_list], dim=0)
    target_matrix = F.normalize(target_matrix, p=2, dim=1)
    target_index = {entity: idx for idx, entity in enumerate(target_list)}

    ranks: List[int] = []
    for source, target in pairs:
        if source not in source_embeddings or target not in target_index:
            continue
        source_vec = source_embeddings[source]
        source_vec = source_vec / (source_vec.norm(p=2) + 1e-12)
        scores = torch.matmul(target_matrix, source_vec)
        _, indices = torch.sort(scores, descending=True)
        desired_idx = target_index[target]
        match = (indices == desired_idx).nonzero(as_tuple=False)
        rank = int(match[0].item()) if match.numel() else len(target_list)
        ranks.append(rank)

    return ranks


def _merge_ranks(*rank_sets: Sequence[int]) -> List[int]:
    merged: List[int] = []
    for ranks in rank_sets:
        merged.extend(ranks)
    return merged


def _ranks_to_metrics(ranks: Sequence[int]) -> Tuple[float, float, float]:
    if not ranks:
        return 0.0, 0.0, 0.0
    total = len(ranks)
    hits1 = sum(1 for rank in ranks if rank == 0) / total
    hits10 = sum(1 for rank in ranks if rank < 10) / total
    mrr = sum(1.0 / (rank + 1) for rank in ranks) / total
    return hits1, hits10, mrr


def compute_basic_unit_metrics(
    model: BasicBertUnitModel,
    data: PreparedBertIntData,
    *,
    device: torch.device,
    batch_size: int,
    eval_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, float]:
    if not eval_pairs:
        logger.warning("[BERT-INT] Evaluation skipped — no pairs available.")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hits@1": 0.0,
            "hits@10": 0.0,
            "mrr": 0.0,
        }

    embeddings_left = _compute_embeddings(
        model,
        data.left_entities,
        data.entity_inputs,
        device,
        batch_size,
    )
    embeddings_right = _compute_embeddings(
        model,
        data.right_entities,
        data.entity_inputs,
        device,
        batch_size,
    )

    ranks_lr = _compute_directional_ranks(
        eval_pairs,
        embeddings_left,
        embeddings_right,
        data.right_entities,
    )
    ranks_rl = _compute_directional_ranks(
        [(tgt, src) for src, tgt in eval_pairs],
        embeddings_right,
        embeddings_left,
        data.left_entities,
    )

    if ranks_lr:
        hits1, hits10, mrr = _ranks_to_metrics(ranks_lr)
        logger.info(
            "[BERT-INT] Evaluation (left→right): hits@1=%.3f hits@10=%.3f mrr=%.3f (pairs=%d)",
            hits1,
            hits10,
            mrr,
            len(ranks_lr),
        )
    if ranks_rl:
        hits1, hits10, mrr = _ranks_to_metrics(ranks_rl)
        logger.info(
            "[BERT-INT] Evaluation (right→left): hits@1=%.3f hits@10=%.3f mrr=%.3f (pairs=%d)",
            hits1,
            hits10,
            mrr,
            len(ranks_rl),
        )

    combined = _merge_ranks(ranks_lr, ranks_rl)
    hits1, hits10, mrr = _ranks_to_metrics(combined)
    logger.info(
        "[BERT-INT] Evaluation (combined): hits@1=%.3f hits@10=%.3f mrr=%.3f (pairs=%d)",
        hits1,
        hits10,
        mrr,
        len(combined),
    )
    return {
        "precision": hits1,
        "recall": hits1,
        "f1": hits1,
        "hits@1": hits1,
        "hits@10": hits10,
        "mrr": mrr,
    }


@MODEL_REGISTRY.register("bert_int")
class BertIntBasicUnit:
    """Coordinate Basic BERT Unit training and interaction evaluation."""

    def __init__(self, stage_config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = stage_config or {}
        overrides = self.stage_config.get("models", {}).get("bert_int", {})
        self.config: BertIntConfig = load_bert_int_config(overrides)
        logger.info(
            "[BERT-INT] Initialising basic unit (encoder=%s, device=%s, pretrained=%s)",
            self.config.basic_unit.encoder_name,
            self.config.device,
            self.config.basic_unit.load_pretrained,
        )
        self.encoder = BasicBertUnitModel(
            input_size=DEFAULT_INPUT_SIZE,
            result_size=DEFAULT_RESULT_SIZE,
            pretrained_model=self.config.basic_unit.encoder_name,
            dropout=self.config.basic_unit.dropout,
            device=self.config.device,
            load_pretrained=self.config.basic_unit.load_pretrained,
        )

    def evaluate(self, dataset_reduced, dataset_augmented) -> Dict[str, float]:
        dataset = dataset_augmented or dataset_reduced
        if dataset is None:
            logger.warning("[BERT-INT] No dataset provided, returning empty metrics.")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "hits@1": 0.0,
                "hits@10": 0.0,
                "mrr": 0.0,
            }

        logger.info("[BERT-INT] Preparing data for Basic Unit training (ratio=%.2f)", self.config.basic_unit.train_ratio)
        prepared_data: PreparedBertIntData = prepare_basic_unit_data(
            dataset,
            train_ratio=self.config.basic_unit.train_ratio,
            max_length=self.config.basic_unit.max_seq_length,
            tokenizer_name=self.config.basic_unit.encoder_name,
            seed=self.config.seed,
        )

        logger.info("[BERT-INT] === Stage 1: Basic Unit fine-tuning ===")
        artifacts: BasicUnitArtifacts = train_basic_unit_model(
            self.encoder,
            prepared_data,
            self.config.basic_unit,
            device=self.config.device,
            cache_dir=self.config.paths.cache_dir,
            seed=self.config.seed,
        )
        logger.info(
            "[BERT-INT] Basic Unit training finished (epochs=%d, final_loss=%.4f)",
            artifacts.epochs,
            artifacts.losses[-1] if artifacts.losses else float("nan"),
        )
        logger.info("[BERT-INT] Optimiser updates executed: %d", artifacts.total_updates)

        eval_pairs = prepared_data.test_pairs
        if not eval_pairs:
            logger.warning(
                "[BERT-INT] No held-out alignment pairs available; evaluating on training pairs."
            )
            eval_pairs = prepared_data.train_pairs

        metrics = compute_basic_unit_metrics(
            self.encoder,
            prepared_data,
            device=self.config.device,
            batch_size=self.config.basic_unit.eval_batch_size,
            eval_pairs=eval_pairs,
        )
        return metrics
