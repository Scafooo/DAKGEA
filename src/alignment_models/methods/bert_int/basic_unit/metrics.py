"""Evaluation helpers for the BERT-INT basic unit."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

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
    left_list = list(embeddings_left)
    right_list = list(embeddings_right)

    if not left_list or not right_list:
        return torch.empty((0, 0), dtype=torch.float32)

    if isinstance(left_list[0], torch.Tensor):
        left_tensor = torch.stack(left_list).to(torch.float32)
    else:
        left_tensor = torch.from_numpy(np.asarray(left_list, dtype=np.float32))

    if isinstance(right_list[0], torch.Tensor):
        right_tensor = torch.stack(right_list).to(torch.float32)
    else:
        right_tensor = torch.from_numpy(np.asarray(right_list, dtype=np.float32))

    device_t = _resolve_device(device)
    left_norm = torch.nn.functional.normalize(left_tensor, p=2, dim=1)
    right_norm = torch.nn.functional.normalize(right_tensor, p=2, dim=1)

    # Move right matrix to device ONCE before loop
    right_norm_t = right_norm.t().to(device_t)

    result_chunks: List[torch.Tensor] = []
    for start in range(0, left_norm.size(0), batch_size):
        chunk = left_norm[start : start + batch_size].to(device_t)
        prod = chunk @ right_norm_t
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
        # Ensure k does not exceed the number of columns (replicate original behavior)
        k = min(topk, batch.size(1))
        batch_scores, batch_indices = batch.topk(k, largest=True)
        scores.append(batch_scores.cpu())
        indices.append(batch_indices.cpu())
    return torch.cat(scores, dim=0), torch.cat(indices, dim=0)


def compute_hits(topk_indices: torch.Tensor) -> dict[str, float]:
    """Compute standard hit@k, mean rank, MRR, and classification metrics."""
    hits = [0.0 for _ in range(topk_indices.size(1))]
    mr = 0.0
    mrr = 0.0
    evaluated = 0

    # For precision/recall/f-measure computation
    true_positives = 0  # Correctly predicted alignments (hits@1)
    false_positives = 0  # Incorrectly predicted alignments
    false_negatives = 0  # Missed alignments

    for row, indices in enumerate(topk_indices):
        found = False
        for rank, candidate in enumerate(indices.tolist()):
            if candidate == row:
                mr += rank + 1
                mrr += 1.0 / (rank + 1)
                evaluated += 1
                for k in range(rank, len(hits)):
                    hits[k] += 1.0

                # Classification metrics: consider hits@1 as true positive
                if rank == 0:
                    true_positives += 1
                else:
                    false_negatives += 1  # Found but not at rank 1
                found = True
                break

        if not found:
            false_negatives += 1

        # Top-1 prediction that's wrong is a false positive
        if len(indices) > 0 and indices[0].item() != row:
            false_positives += 1

    total = topk_indices.size(0)
    if total == 0:
        return {
            "hits@1": 0.0,
            "hits@5": 0.0,
            "hits@10": 0.0,
            "mr": 0.0,
            "mrr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f-measure": 0.0,
            "evaluated": 0,
        }

    hits = [value / total for value in hits]

    # Compute precision, recall, and f-measure
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f_measure = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "hits@1": hits[0] if len(hits) >= 1 else 0.0,
        "hits@5": hits[4] if len(hits) >= 5 else hits[-1],
        "hits@10": hits[9] if len(hits) >= 10 else hits[-1],
        "mr": mr / max(evaluated, 1),
        "mrr": mrr / max(evaluated, 1),
        "precision": precision,
        "recall": recall,
        "f-measure": f_measure,
        "evaluated": evaluated,
    }
