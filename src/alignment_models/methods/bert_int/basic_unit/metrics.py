"""Evaluation helpers for the BERT-INT basic unit."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch


def _resolve_device(device: Optional[int | str | torch.device]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def batch_cosine_similarity(
    embeddings_left: Iterable,
    embeddings_right: Iterable,
    batch_size: int,
    device: Optional[int | str | torch.device] = None,
) -> torch.Tensor:
    """Return the cosine similarity matrix computed in mini-batches."""
    left_tensor = torch.as_tensor(list(embeddings_left), dtype=torch.float32)
    right_tensor = torch.as_tensor(list(embeddings_right), dtype=torch.float32)

    device_t = _resolve_device(device)
    left_norm = torch.nn.functional.normalize(left_tensor, p=2, dim=1)
    right_norm = torch.nn.functional.normalize(right_tensor, p=2, dim=1)

    result_chunks: List[torch.Tensor] = []
    for start in range(0, left_norm.size(0), batch_size):
        chunk = left_norm[start : start + batch_size].to(device_t)
        prod = chunk @ right_norm.t().to(device_t)
        result_chunks.append(prod.cpu())
    return torch.cat(result_chunks, dim=0)


def batch_topk(
    matrix: torch.Tensor,
    batch_size: int,
    topk: int,
    device: Optional[int | str | torch.device] = None,
) -> torch.return_types.topk:
    """Return batched top-k scores and indices."""
    device_t = _resolve_device(device)
    scores: List[torch.Tensor] = []
    indices: List[torch.Tensor] = []
    for start in range(0, matrix.size(0), batch_size):
        batch = matrix[start : start + batch_size].to(device_t)
        batch_scores, batch_indices = batch.topk(topk, largest=True)
        scores.append(batch_scores.cpu())
        indices.append(batch_indices.cpu())
    return torch.cat(scores, dim=0), torch.cat(indices, dim=0)


def compute_hits(topk_indices: torch.Tensor) -> dict[str, float]:
    """Compute standard hit@k, mean rank, and MRR metrics."""
    hits = [0.0 for _ in range(topk_indices.size(1))]
    mr = 0.0
    mrr = 0.0
    evaluated = 0

    for row, indices in enumerate(topk_indices):
        for rank, candidate in enumerate(indices.tolist()):
            if candidate == row:
                mr += rank + 1
                mrr += 1.0 / (rank + 1)
                evaluated += 1
                for k in range(rank, len(hits)):
                    hits[k] += 1.0
                break

    total = topk_indices.size(0)
    if total == 0:
        return {
            "hits@1": 0.0,
            "hits@5": 0.0,
            "hits@10": 0.0,
            "mr": 0.0,
            "mrr": 0.0,
            "evaluated": 0,
        }

    hits = [value / total for value in hits]
    return {
        "hits@1": hits[0] if len(hits) >= 1 else 0.0,
        "hits@5": hits[4] if len(hits) >= 5 else hits[-1],
        "hits@10": hits[9] if len(hits) >= 10 else hits[-1],
        "mr": mr / max(evaluated, 1),
        "mrr": mrr / max(evaluated, 1),
        "evaluated": evaluated,
    }
