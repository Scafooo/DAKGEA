"""Greedy alignment evaluation (hits@k, MRR) for MultiKE."""

from __future__ import annotations

import gc
import multiprocessing

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances


def sim_matrix(embed1, embed2, metric='inner', normalize=False):
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        return np.matmul(embed1, embed2.T).astype(np.float32)
    elif metric == 'euclidean':
        return (1 - euclidean_distances(embed1, embed2)).astype(np.float32)
    else:
        return (1 - cdist(embed1, embed2, metric=metric)).astype(np.float32)


def _calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    mr = mrr = 0.0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        rank_index = int(np.where(rank == gold)[0][0])
        mr += rank_index + 1
        mrr += 1.0 / (rank_index + 1)
        for j, k in enumerate(top_k):
            if rank_index < k:
                hits[j] += 1
    return mr / total_num, mrr / total_num, hits, hits1_rest


def greedy_alignment(embed1, embed2, top_k, threads_num, metric='inner', normalize=False, accurate=False):
    s = sim_matrix(embed1, embed2, metric=metric, normalize=normalize)
    num = s.shape[0]

    if threads_num > 1:
        tasks = _task_divide(list(range(num)), threads_num)
        pool = multiprocessing.Pool(processes=len(tasks))
        rests = [pool.apply_async(_calculate_rank, (task, s[task, :], top_k, accurate, num))
                 for task in tasks]
        pool.close()
        pool.join()
        mr = mrr = 0.0
        hits = [0] * len(top_k)
        alignment_rest = set()
        for res in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1 = res.get()
            mr += sub_mr * (len(tasks[0]) / num)  # weighted average
            mrr += sub_mrr * (len(tasks[0]) / num)
            hits = [h + sh for h, sh in zip(hits, sub_hits)]
            alignment_rest |= sub_hits1
    else:
        mr, mrr, hits, alignment_rest = _calculate_rank(list(range(num)), s, top_k, accurate, num)

    hits_pct = [round(h / num * 100, 3) for h in hits]
    del s
    gc.collect()
    return alignment_rest, hits_pct, mr, mrr


def _task_divide(idx, n):
    total = len(idx)
    if n <= 0 or total == 0 or n > total:
        return [idx]
    j = total // n
    tasks = [idx[i:i + j] for i in range(0, (n - 1) * j, j)]
    tasks.append(idx[(n - 1) * j:])
    return tasks


def evaluate(embeds_src, embeds_tgt, top_k=(1, 5, 10), normalize=True):
    """Return hits@k dict and mrr for source → target alignment."""
    _, hits_pct, _, mrr = greedy_alignment(
        embeds_src, embeds_tgt, list(top_k), threads_num=1, normalize=normalize, accurate=True)
    result = {f"hits@{k}": h / 100.0 for k, h in zip(top_k, hits_pct)}
    result["mrr"] = mrr
    return result
