"""Utility helpers for similarity computations."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(
    emb1,
    emb2,
    *,
    batch_size: int = 512,
    device: torch.device,
) -> torch.Tensor:
    """Compute cosine similarity matrix between two embedding collections."""

    a = torch.tensor(emb1, dtype=torch.float32, device=device)
    b = torch.tensor(emb2, dtype=torch.float32, device=device)
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    chunks: Tuple[torch.Tensor, ...] = tuple(
        a[i : i + batch_size] @ b.t()
        for i in range(0, a.size(0), batch_size)
    )
    return torch.cat(chunks, dim=0)


def batched_topk(
    matrix: torch.Tensor,
    *,
    k: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k selection that mirrors the GPU friendly original implementation."""

    scores: Tuple[torch.Tensor, ...] = ()
    indices: Tuple[torch.Tensor, ...] = ()
    for start in range(0, matrix.size(0), batch_size):
        chunk = matrix[start : start + batch_size].to(device)
        score, index = chunk.topk(k, largest=True)
        if scores:
            scores += (score.cpu(),)
            indices += (index.cpu(),)
        else:
            scores = (score.cpu(),)
            indices = (index.cpu(),)
    return torch.cat(scores, dim=0), torch.cat(indices, dim=0)
