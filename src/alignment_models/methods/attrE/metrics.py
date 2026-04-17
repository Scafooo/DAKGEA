"""Evaluation metrics for AttrE."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

Pair = Tuple[int, int]


def compute_metrics(
    emb_kg1: torch.Tensor,
    emb_kg2: torch.Tensor,
    test_pairs: List[Pair],
    kg1_entity_ids: List[int],
    kg2_entity_ids: List[int],
    batch_size: int = 512,
) -> Dict[str, float]:
    """Compute hits@1, hits@10 and MRR for entity alignment.

    Evaluation direction: for each test pair (e1_global, e2_global), rank
    all KG1 entities by cosine similarity to e2, and check where e1 lands.

    Args:
        emb_kg1: ``[|KG1|, H]`` L2-normalised relation embeddings for KG1.
        emb_kg2: ``[|KG2|, H]`` L2-normalised relation embeddings for KG2.
        test_pairs: List of (global_e1_id, global_e2_id) test alignment pairs.
        kg1_entity_ids: Ordered list of global entity IDs in KG1 — defines the
            rows of *emb_kg1* (row i corresponds to kg1_entity_ids[i]).
        kg2_entity_ids: Same for KG2 / *emb_kg2*.
        batch_size: Rows of emb_kg2 processed at once (memory control).

    Returns:
        ``{"hits@1": float, "hits@10": float, "mrr": float}``
    """
    if not test_pairs:
        return {"hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

    # Maps global entity ID → position in the embedding matrix
    kg1_id2pos = {eid: i for i, eid in enumerate(kg1_entity_ids)}
    kg2_id2pos = {eid: i for i, eid in enumerate(kg2_entity_ids)}

    # Filter pairs to those present in both embedding matrices
    valid_pairs = [
        (e1, e2)
        for e1, e2 in test_pairs
        if e1 in kg1_id2pos and e2 in kg2_id2pos
    ]
    if not valid_pairs:
        return {"hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

    emb_kg1 = emb_kg1.float()
    emb_kg2 = emb_kg2.float()

    hits1 = hits10 = 0
    mrr_sum = 0.0

    for start in range(0, len(valid_pairs), batch_size):
        batch = valid_pairs[start : start + batch_size]
        e2_pos = [kg2_id2pos[e2] for _, e2 in batch]
        e1_pos = [kg1_id2pos[e1] for e1, _ in batch]

        q = emb_kg2[e2_pos]          # [B, H]
        sims = q @ emb_kg1.T         # [B, |KG1|]

        for i, target_pos in enumerate(e1_pos):
            row = sims[i]
            # rank among KG1 entities (0-based, lower = better)
            rank = int((row > row[target_pos]).sum().item())

            if rank == 0:
                hits1 += 1
            if rank < 10:
                hits10 += 1
            mrr_sum += 1.0 / (rank + 1)

    total = len(valid_pairs)
    return {
        "hits@1": hits1 / total,
        "hits@10": hits10 / total,
        "mrr": mrr_sum / total,
    }
