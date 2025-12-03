"""Utility functions for RREA model."""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Set, Dict, Optional
from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Normalize adjacency matrix using symmetric normalization.

    Args:
        adj: Sparse adjacency matrix

    Returns:
        Normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def get_matrix_from_dataset(
    dataset: Dataset,
    train_pairs: np.ndarray
) -> Tuple[sp.spmatrix, np.ndarray, np.ndarray, sp.spmatrix, sp.spmatrix, Dict[str, int], int]:
    """Build adjacency matrices and relation features from dataset.

    Args:
        dataset: DAKGEA Dataset object
        train_pairs: Training alignment pairs

    Returns:
        Tuple of (adj_matrix, r_index, r_val, adj_features, rel_features, entity2id, kg2_offset)
    """
    from rdflib import Literal

    # Build entity and relation mappings dynamically
    all_entities = set()
    all_relations = set([0])  # Add dummy relation 0
    entity2id = {}
    relation2id = {"_dummy_": 0}  # Reserve 0 for dummy
    triples = []

    # Process source KG - build entity mapping
    kg1 = dataset.knowledge_graph_source
    for s, p, o in kg1:
        s_str = str(s)
        o_str = str(o)

        # Only process relation triples (not attribute triples)
        if isinstance(o, Literal):
            continue

        # Add entities to mapping
        if s_str not in entity2id:
            entity2id[s_str] = len(entity2id)
        if o_str not in entity2id:
            entity2id[o_str] = len(entity2id)

    # Process target KG - continue entity mapping
    kg2_offset = len(entity2id)
    kg2 = dataset.knowledge_graph_target
    for s, p, o in kg2:
        s_str = str(s)
        o_str = str(o)

        # Only process relation triples (not attribute triples)
        if isinstance(o, Literal):
            continue

        # Add entities to mapping
        if s_str not in entity2id:
            entity2id[s_str] = len(entity2id)
        if o_str not in entity2id:
            entity2id[o_str] = len(entity2id)

    # Process source KG triples
    for s, p, o in kg1:
        s_str = str(s)
        o_str = str(o)
        p_str = str(p)

        if isinstance(o, Literal):
            continue

        s_id = entity2id[s_str]
        o_id = entity2id[o_str]

        # Get or create relation ID
        if p_str not in relation2id:
            relation2id[p_str] = len(relation2id)
        rel_id = relation2id[p_str]

        all_entities.add(s_id)
        all_entities.add(o_id)
        all_relations.add(rel_id)
        triples.append((s_id, rel_id, o_id))

    # Process target KG triples
    for s, p, o in kg2:
        s_str = str(s)
        o_str = str(o)
        p_str = str(p)

        if isinstance(o, Literal):
            continue

        s_id = entity2id[s_str]
        o_id = entity2id[o_str]

        # Get or create relation ID
        if p_str not in relation2id:
            relation2id[p_str] = len(relation2id)
        rel_id = relation2id[p_str]

        all_entities.add(s_id)
        all_entities.add(o_id)
        all_relations.add(rel_id)
        triples.append((s_id, rel_id, o_id))

    # Build matrices
    ent_size = max(all_entities) + 1
    rel_size = max(all_relations) + 1

    logger.info(f"[RREA] Building matrices: entities={ent_size}, relations={rel_size}, triples={len(triples)}")

    # Initialize matrices
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    # Add self-loops
    for i in range(ent_size):
        adj_features[i, i] = 1

    # Process triples
    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r])
        radj.append([t, h, r + rel_size])
        rel_out[h][r] += 1
        rel_in[t][r] += 1

    # Build relation indices and values
    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []

    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        key = f"{h} {t}"
        if key in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(key)
            r_index.append([count, r])
            r_val.append(1)

    # Normalize relation values
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    # Concatenate relation features
    rel_features = np.concatenate([rel_in, rel_out], axis=1)

    # Normalize matrices
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))

    return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features, entity2id, kg2_offset


def build_matrices_from_triples(
    triples: np.ndarray,
    num_entities: int,
    num_relations: int,
) -> Tuple[sp.spmatrix, np.ndarray, np.ndarray, sp.spmatrix, sp.spmatrix, np.ndarray]:
    """Build adjacency matrices from pre-processed triples.

    This function is optimized for pre-processed data where entity and relation
    IDs are already assigned (e.g., from OpenEA writer).

    Args:
        triples: Numpy array of triples [num_triples, 3] with (h, r, t)
        num_entities: Total number of entities
        num_relations: Total number of relations

    Returns:
        Tuple of (adj_matrix, r_index, r_val, adj_features, rel_features, adj_indices)
        where adj_indices contains all directed edges matching r_index
    """
    logger.info(f"[RREA] Building matrices from {len(triples)} triples")
    logger.info(f"[RREA] Entities: {num_entities}, Relations: {num_relations}")

    # Initialize matrices
    adj_matrix = sp.lil_matrix((num_entities, num_entities))
    adj_features = sp.lil_matrix((num_entities, num_entities))
    radj = []
    rel_in = np.zeros((num_entities, num_relations))
    rel_out = np.zeros((num_entities, num_relations))

    # Add self-loops
    for i in range(num_entities):
        adj_features[i, i] = 1

    # Process triples
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)

        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r])
        radj.append([t, h, r + num_relations])
        rel_out[h][r] += 1
        rel_in[t][r] += 1

    # Build relation indices and values
    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []

    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        key = f"{h} {t}"
        if key in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(key)
            r_index.append([count, r])
            r_val.append(1)

    # Normalize relation values
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    # Concatenate relation features
    rel_features = np.concatenate([rel_in, rel_out], axis=1)

    # Build adj_indices from radj to match r_index structure
    # radj contains all directed edges with their relations
    adj_indices_list = [[h, t] for h, t, r in radj]
    adj_indices = np.array(adj_indices_list, dtype=np.int32)

    # Normalize matrices
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))

    logger.info(f"[RREA] Matrix construction completed (adj_indices: {len(adj_indices)} edges)")

    return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features, adj_indices


def get_hits(
    vec: np.ndarray,
    test_pair: np.ndarray,
    wrank: np.ndarray = None,
    top_k: Tuple[int, ...] = (1, 5, 10, 50)
) -> Dict[str, float]:
    """Calculate Hits@K and MRR metrics.

    Args:
        vec: Entity embeddings
        test_pair: Test alignment pairs
        wrank: Optional weighted rank
        top_k: Top-K values to evaluate

    Returns:
        Dictionary with metric results
    """
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])

    # Normalize vectors
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)

    # Compute similarity matrix
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)

    # Apply weighted rank if provided
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i, sim[i, j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank, -1), np.expand_dims(wrank, -1)], -1), axis=-1)
        sim = rank.argsort(-1)

    # Calculate metrics for left-to-right
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    # Calculate metrics for right-to-left
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    # Prepare results
    results = {}
    logger.info("[RREA] Evaluation results (Left to Right):")
    for i in range(len(top_lr)):
        hits = top_lr[i] / len(test_pair) * 100
        results[f"hits@{top_k[i]}_lr"] = hits
        logger.info(f"  Hits@{top_k[i]}: {hits:.2f}%")

    mrr_lr = MRR_lr / Lvec.shape[0]
    results["mrr_lr"] = mrr_lr
    logger.info(f"  MRR: {mrr_lr:.3f}")

    logger.info("[RREA] Evaluation results (Right to Left):")
    for i in range(len(top_rl)):
        hits = top_rl[i] / len(test_pair) * 100
        results[f"hits@{top_k[i]}_rl"] = hits
        logger.info(f"  Hits@{top_k[i]}: {hits:.2f}%")

    mrr_rl = MRR_rl / Rvec.shape[0]
    results["mrr_rl"] = mrr_rl
    logger.info(f"  MRR: {mrr_rl:.3f}")

    # Average results
    for i in range(len(top_k)):
        results[f"hits@{top_k[i]}"] = (results[f"hits@{top_k[i]}_lr"] + results[f"hits@{top_k[i]}_rl"]) / 2
    results["mrr"] = (mrr_lr + mrr_rl) / 2

    return results
