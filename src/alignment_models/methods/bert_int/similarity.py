"""Similarity helpers mirroring the original BERT-INT implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(emb1, emb2, batch_size: int = 128, device: torch.device | None = None):
    vec1 = F.normalize(torch.tensor(emb1, dtype=torch.float32), p=2, dim=1)
    vec2 = F.normalize(torch.tensor(emb2, dtype=torch.float32), p=2, dim=1)
    return _batched_matmul(vec1, vec2.t(), batch_size=batch_size, device=device)


def _batched_matmul(mat1, mat2, batch_size: int, device: torch.device | None = None):
    device = device or torch.device("cpu")
    results = []
    for start in range(0, mat1.shape[0], batch_size):
        chunk = mat1[start : start + batch_size].to(device)
        res = chunk.mm(mat2.to(device))
        results.append(res.cpu())
    return torch.cat(results, dim=0)


def batched_topk(matrix, k: int = None, topn: int = None, batch_size: int = 128, largest: bool = True, device: torch.device | None = None):
    """
    Compute top-k elements for each row of a matrix in batches.

    Args:
        matrix: 2D tensor of shape [n_rows, n_cols]
        k: number of top elements to return (preferred parameter)
        topn: number of top elements to return (legacy parameter, alternative to k)
        batch_size: batch size for processing
        largest: if True, return largest elements; if False, return smallest
        device: device to use for computation

    Returns:
        Tuple of (scores, indices) tensors
    """
    # Handle both k and topn parameters for backward compatibility
    if k is None and topn is None:
        raise ValueError("Either 'k' or 'topn' must be specified")
    if k is None:
        k = topn

    device = device or torch.device("cpu")

    # Ensure k doesn't exceed the number of columns in the matrix
    k_actual = min(k, matrix.shape[1]) if matrix.shape[1] > 0 else 0

    if k_actual == 0:
        # Return empty tensors with correct shape
        return torch.empty((matrix.shape[0], 0)), torch.empty((matrix.shape[0], 0), dtype=torch.long)

    scores = []
    indices = []
    for start in range(0, matrix.shape[0], batch_size):
        chunk = matrix[start : start + batch_size].to(device)
        score_chunk, index_chunk = torch.topk(chunk, k=k_actual, largest=largest)
        scores.append(score_chunk.cpu())
        indices.append(index_chunk.cpu())

    result_scores = torch.cat(scores, dim=0)
    result_indices = torch.cat(indices, dim=0)
    return result_scores, result_indices
