"""Cross-domain Similarity Local Scaling (CSLS) for entity alignment."""

import numpy as np
import torch
from typing import Tuple, Set
from scipy.spatial.distance import cdist
from src.logger import get_logger

logger = get_logger(__name__)


def csls_sim(
    embed1: np.ndarray,
    embed2: np.ndarray,
    k: int = 10,
    num_threads: int = 16
) -> np.ndarray:
    """Compute CSLS similarity between two sets of embeddings.

    CSLS reduces the hubness problem in nearest neighbor search by
    penalizing entities that are nearest neighbors to many others.

    Args:
        embed1: Source embeddings [n1, d]
        embed2: Target embeddings [n2, d]
        k: Number of nearest neighbors for CSLS
        num_threads: Number of threads (legacy parameter, not used in numpy)

    Returns:
        CSLS similarity matrix [n1, n2]
    """
    # Normalize embeddings
    embed1 = embed1 / (np.linalg.norm(embed1, axis=1, keepdims=True) + 1e-10)
    embed2 = embed2 / (np.linalg.norm(embed2, axis=1, keepdims=True) + 1e-10)

    # Compute cosine similarity
    sim_mat = embed1 @ embed2.T

    # Compute mean similarity of k nearest neighbors
    # For each row in embed1, find k nearest in embed2
    mean_sim1 = np.mean(np.sort(sim_mat, axis=1)[:, -k:], axis=1, keepdims=True)

    # For each column in embed2, find k nearest in embed1
    mean_sim2 = np.mean(np.sort(sim_mat, axis=0)[-k:, :], axis=0, keepdims=True)

    # CSLS similarity
    csls_sim_mat = 2 * sim_mat - mean_sim1 - mean_sim2

    return csls_sim_mat


def eval_alignment_by_sim_mat(
    embed1: np.ndarray,
    embed2: np.ndarray,
    test_pairs: np.ndarray,
    top_k: Tuple[int, ...] = (1, 5, 10, 50),
    csls_k: int = 10,
    use_csls: bool = True,
    num_threads: int = 16,
    accurate: bool = True,
) -> Tuple[Set[Tuple[int, int]], float, dict]:
    """Evaluate alignment using similarity matrix.

    Args:
        embed1: Source entity embeddings
        embed2: Target entity embeddings
        test_pairs: Test alignment pairs [num_pairs, 2]
        top_k: Top-K values to evaluate
        csls_k: K for CSLS
        use_csls: Whether to use CSLS or cosine similarity
        num_threads: Number of threads
        accurate: Whether to compute accurate ranking (vs approximate)

    Returns:
        Tuple of (predicted_pairs, hits@1, all_metrics)
    """
    # Build similarity matrix
    if use_csls:
        logger.info(f"[STEP] Computing CSLS similarity (k={csls_k})")
        sim_mat = csls_sim(embed1, embed2, k=csls_k, num_threads=num_threads)
    else:
        logger.info("[STEP] Computing cosine similarity")
        embed1 = embed1 / (np.linalg.norm(embed1, axis=1, keepdims=True) + 1e-10)
        embed2 = embed2 / (np.linalg.norm(embed2, axis=1, keepdims=True) + 1e-10)
        sim_mat = embed1 @ embed2.T

    # Compute metrics
    # Note: sim_mat already has rows only for test entities (embed1 was pre-selected)
    metrics = {}
    num = np.zeros(len(top_k), dtype=int)
    mean_rank = 0.0
    mrr = 0.0
    prec_set = set()

    for i, (src_id, tgt_id) in enumerate(test_pairs):
        # Get similarity scores for this source entity (i is the row index)
        scores = sim_mat[i, :]

        # Rank target entities
        if accurate:
            rank = np.argsort(-scores)
        else:
            # Approximate ranking using partition
            rank = np.argpartition(-scores, np.array(top_k) - 1)

        # Find rank of true match
        rank_index = np.where(rank == tgt_id)[0][0]

        # Add to prediction set
        prec_set.add((src_id, rank[0]))

        # Update metrics
        mean_rank += (rank_index + 1)
        mrr += 1.0 / (rank_index + 1)

        for j, k in enumerate(top_k):
            if rank_index < k:
                num[j] += 1

    # Compute final metrics
    n_test = len(test_pairs)
    mean_rank /= n_test
    mrr /= n_test

    for j, k in enumerate(top_k):
        hits = num[j] / n_test * 100
        metrics[f"hits@{k}"] = hits
        logger.info(f"[RREA] Hits@{k}: {hits:.2f}%")

    metrics["mean_rank"] = mean_rank
    metrics["mrr"] = mrr
    logger.info(f"[RREA] Mean Rank: {mean_rank:.2f}")
    logger.info(f"[RREA] MRR: {mrr:.4f}")

    hits1 = metrics["hits@1"]

    return prec_set, hits1, metrics


def eval_alignment_batched(
    model: torch.nn.Module,
    test_pairs: np.ndarray,
    adj_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_val: torch.Tensor,
    kg2_offset: int,
    top_k: Tuple[int, ...] = (1, 5, 10, 50),
    csls_k: int = 10,
    use_csls: bool = True,
    device: str = "cuda",
) -> dict:
    """Evaluate alignment using batched forward pass through model.

    Args:
        model: RREA encoder model
        test_pairs: Test alignment pairs
        adj_indices: Adjacency indices
        sparse_indices: Sparse relation indices
        sparse_val: Sparse relation values
        kg2_offset: Offset for KG2 entity IDs
        top_k: Top-K values
        csls_k: K for CSLS
        use_csls: Whether to use CSLS
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model.eval()
    with torch.no_grad():
        # Get all entity embeddings
        embeddings = model(adj_indices, sparse_indices, sparse_val)
        embeddings = embeddings.cpu().numpy()

    # Split into KG1 and KG2 embeddings
    embed1 = embeddings[:kg2_offset]
    embed2 = embeddings[kg2_offset:]

    # Adjust test pairs for KG2 offset
    test_pairs_adjusted = test_pairs.copy()
    test_pairs_adjusted[:, 1] -= kg2_offset

    # Evaluate
    _, hits1, metrics = eval_alignment_by_sim_mat(
        embed1=embed1[test_pairs_adjusted[:, 0]],
        embed2=embed2,
        test_pairs=test_pairs_adjusted,
        top_k=top_k,
        csls_k=csls_k,
        use_csls=use_csls,
    )

    return metrics
