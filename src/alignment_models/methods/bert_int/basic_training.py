"""Basic BERT unit training adapted from the original BERT-INT code base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from src.logger import get_logger

from .batch_generator import BatchTrainDataGenerator
from .similarity import batched_topk, cosine_similarity_matrix

logger = get_logger(__name__)


Pair = Tuple[int, int]


@dataclass
class BasicUnitArtifacts:
    entity_embeddings: Sequence[Sequence[float]]
    train_pairs: Sequence[Pair]
    test_pairs: Sequence[Pair]
    train_candidates: Dict[int, Sequence[int]]
    test_candidates: Dict[int, Sequence[int]]
    entity_pairs: Sequence[Pair]
    entid2data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]


def _entlist_to_embeddings(
    model: nn.Module,
    entids: Sequence[int],
    entid2data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    requires_grad: bool = False,
    batch_size: int = 256,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    effective_batch = batch_size if batch_size > 0 else len(entids)
    for start in range(0, len(entids), effective_batch):
        chunk = entids[start : start + effective_batch]
        if not chunk:
            continue
        token_ids = torch.stack([entid2data[e][0] for e in chunk]).to(device)
        mask_ids = torch.stack([entid2data[e][1] for e in chunk]).to(device)
        if requires_grad:
            outputs.append(model(token_ids, mask_ids))
        else:
            with torch.no_grad():
                outputs.append(model(token_ids, mask_ids))
    if not outputs:
        out_dim = getattr(getattr(model, "output_layer", None), "out_features", 0)
        return torch.empty((0, out_dim), device=device)
    return torch.cat(outputs, dim=0)


def _generate_candidate_dict(
    model: nn.Module,
    entid2data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    train_ids1: Sequence[int],
    train_ids2: Sequence[int],
    pool_ids1: Sequence[int],
    pool_ids2: Sequence[int],
    topk: int,
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[int, list[int]]:
    model.eval()
    logger.debug(
        "[BERT-INT] Generating candidate dictionary (|train_ids1|=%d, |train_ids2|=%d, pool1=%d, pool2=%d)",
        len(train_ids1),
        len(train_ids2),
        len(pool_ids1),
        len(pool_ids2),
    )
    with torch.no_grad():
        emb1 = _entlist_to_embeddings(
            model,
            pool_ids1,
            entid2data,
            device,
            batch_size=batch_size,
        )
        emb2 = _entlist_to_embeddings(
            model,
            pool_ids2,
            entid2data,
            device,
            batch_size=batch_size,
        )
    emb1 = emb1.cpu()
    emb2 = emb2.cpu()

    pool_index1 = {eid: i for i, eid in enumerate(pool_ids1)}
    pool_index2 = {eid: i for i, eid in enumerate(pool_ids2)}
    train_emb1 = emb1[[pool_index1[idx] for idx in train_ids1]].numpy().tolist()
    train_emb2 = emb2[[pool_index2[idx] for idx in train_ids2]].numpy().tolist()
    pool_emb1 = emb1.numpy().tolist()
    pool_emb2 = emb2.numpy().tolist()

    sim1 = cosine_similarity_matrix(train_emb1, pool_emb2, batch_size=2048, device=device)
    _, idx1 = batched_topk(sim1, k=topk, batch_size=2048, device=device)
    sim2 = cosine_similarity_matrix(train_emb2, pool_emb1, batch_size=2048, device=device)
    _, idx2 = batched_topk(sim2, k=topk, batch_size=2048, device=device)

    candidate_dict: Dict[int, list[int]] = {}
    for row, e1 in enumerate(train_ids1):
        candidate_dict[e1] = [pool_ids2[i] for i in idx1[row].tolist()]
    for row, e2 in enumerate(train_ids2):
        candidate_dict[e2] = [pool_ids1[i] for i in idx2[row].tolist()]

    avg_cands = float(sum(len(v) for v in candidate_dict.values())) / max(1, len(candidate_dict))
    logger.info(
        "[BERT-INT] Candidate dict: %d entities, avg %.2f candidates per entity, topk=%d",
        len(candidate_dict),
        avg_cands,
        topk,
    )
    return candidate_dict


def _margin_ranking_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_pe1: Sequence[int],
    batch_pe2: Sequence[int],
    batch_ne1: Sequence[int],
    batch_ne2: Sequence[int],
    entid2data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    embedding_batch_size: int,
) -> float:
    model.train()
    optimizer.zero_grad()

    pos_emb1 = _entlist_to_embeddings(
        model,
        batch_pe1,
        entid2data,
        device,
        requires_grad=True,
        batch_size=embedding_batch_size,
    )
    pos_emb2 = _entlist_to_embeddings(
        model,
        batch_pe2,
        entid2data,
        device,
        requires_grad=True,
        batch_size=embedding_batch_size,
    )
    neg_emb1 = _entlist_to_embeddings(
        model,
        batch_ne1,
        entid2data,
        device,
        requires_grad=True,
        batch_size=embedding_batch_size,
    )
    neg_emb2 = _entlist_to_embeddings(
        model,
        batch_ne2,
        entid2data,
        device,
        requires_grad=True,
        batch_size=embedding_batch_size,
    )

    pos_score = F.pairwise_distance(pos_emb1, pos_emb2, p=1).unsqueeze(-1)
    neg_score = F.pairwise_distance(neg_emb1, neg_emb2, p=1).unsqueeze(-1)

    label = -torch.ones_like(pos_score)
    loss = criterion(pos_score.to(device), neg_score.to(device), label.to(device))
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_basic_unit_model(
    model: nn.Module,
    entid2data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    ent_ids1: Sequence[int],
    ent_ids2: Sequence[int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    negatives: int,
    candidate_topk: int,
    eval_topk: int,
    device: torch.device,
    embedding_batch_size: int,
) -> BasicUnitArtifacts:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=margin)

    generator = BatchTrainDataGenerator(
        train_pairs,
        ent_ids1,
        ent_ids2,
        {idx: str(idx) for idx in range(len(entid2data))},
        batch_size=batch_size,
        neg_num=negatives,
    )

    dummy_candidates: Dict[int, Sequence[int]] = {}
    ent_ids1_list = list(ent_ids1)
    ent_ids2_list = list(ent_ids2)
    for idx in ent_ids1_list:
        candidates = ent_ids2_list if ent_ids2_list else [idx]
        dummy_candidates[idx] = candidates
    for idx in ent_ids2_list:
        candidates = ent_ids1_list if ent_ids1_list else [idx]
        dummy_candidates[idx] = candidates
    if not dummy_candidates:
        raise ValueError("No training entities available to build negative samples.")
    generator.train_index_gene(dummy_candidates)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for pe1s, pe2s, ne1s, ne2s in generator:
            loss = _margin_ranking_step(
                model,
                criterion,
                optimizer,
                pe1s,
                pe2s,
                ne1s,
                ne2s,
                entid2data,
                device,
                embedding_batch_size=embedding_batch_size,
            )
            epoch_loss += loss
            batches += 1
        logger.info(
            "[BERT-INT] Basic unit epoch %d loss %.4f (batches=%d)",
            epoch + 1,
            epoch_loss / max(1, batches),
            batches,
        )

    with torch.no_grad():
        embeddings = _entlist_to_embeddings(model, sorted(entid2data.keys()), entid2data, device).cpu()
    logger.info("[BERT-INT] Basic unit produced embeddings for %d entities.", embeddings.size(0))

    entity_embeddings = embeddings.numpy().tolist()
    train_candidates = _generate_candidate_dict(
        model,
        entid2data,
        [src for src, _ in train_pairs],
        [tgt for _, tgt in train_pairs],
        ent_ids1,
        ent_ids2,
        candidate_topk,
        device,
        batch_size=embedding_batch_size,
    )
    test_candidates = _generate_candidate_dict(
        model,
        entid2data,
        [src for src, _ in test_pairs],
        [tgt for _, tgt in test_pairs],
        ent_ids1,
        ent_ids2,
        eval_topk,
        device,
        batch_size=embedding_batch_size,
    )

    entity_pairs_set = set()
    for cand in (train_candidates, test_candidates):
        for source, targets in cand.items():
            for target in targets:
                entity_pairs_set.add((source, target))
    entity_pairs_set.update(train_pairs)
    entity_pairs = sorted(entity_pairs_set)
    logger.info(
        "[BERT-INT] Entity pairs: %d (train_candidates=%d, test_candidates=%d, train_pairs=%d)",
        len(entity_pairs),
        len(train_candidates),
        len(test_candidates),
        len(train_pairs),
    )

    return BasicUnitArtifacts(
        entity_embeddings=entity_embeddings,
        train_pairs=list(train_pairs),
        test_pairs=list(test_pairs),
        train_candidates=train_candidates,
        test_candidates=test_candidates,
        entity_pairs=entity_pairs,
        entid2data=entid2data,
    )
