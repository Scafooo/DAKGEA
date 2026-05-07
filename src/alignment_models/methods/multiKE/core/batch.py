"""Batch generation utilities for relation and attribute triple training."""

from __future__ import annotations

import gc
import multiprocessing
import random

import numpy as np


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or total == 0 or n > total:
        return [idx]
    j = total // n
    tasks = [idx[i:i + j] for i in range(0, (n - 1) * j, j)]
    tasks.append(idx[(n - 1) * j:])
    return tasks


def generate_pos_triples(triples, batch_size, step):
    start = step * batch_size
    end = min(start + batch_size, len(triples))
    return triples[start:end]


def generate_neg_triples_fast(pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    if neighbor is None:
        neighbor = {}
    neg_batch = []
    for head, relation, tail in pos_batch:
        neg_triples = []
        nums_to_sample = neg_triples_num
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg)
                break
            i_neg = list(i_neg - all_triples_set)
            neg_triples += i_neg
            if len(neg_triples) == neg_triples_num:
                break
            nums_to_sample = neg_triples_num - len(neg_triples)
        neg_batch.extend(neg_triples)
    return neg_batch


def generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                   entity_list1, entity_list2, batch_size,
                                   step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos2 = generate_pos_triples(triple_list2, batch_size2, step)
    neg1 = generate_neg_triples_fast(pos1, triple_set1, entity_list1, neg_triples_num, neighbor=neighbor1)
    neg2 = generate_neg_triples_fast(pos2, triple_set2, entity_list2, neg_triples_num, neighbor=neighbor2)
    return pos1 + pos2, neg1 + neg2


def _generate_relation_batch_worker(args):
    (triple_list1, triple_list2, triple_set1, triple_set2,
     entity_list1, entity_list2, batch_size, steps, neighbor1, neighbor2, neg_triples_num) = args
    results = []
    for step in steps:
        results.append(generate_relation_triple_batch(
            triple_list1, triple_list2, triple_set1, triple_set2,
            entity_list1, entity_list2, batch_size, step, neighbor1, neighbor2, neg_triples_num))
    return results


def generate_all_relation_batches(triple_list1, triple_list2, triple_set1, triple_set2,
                                  entity_list1, entity_list2, batch_size, triple_steps,
                                  neighbor1, neighbor2, neg_triples_num):
    """Generate all batches for one epoch (no multiprocessing for TF2 compatibility)."""
    batches = []
    for step in range(triple_steps):
        pos, neg = generate_relation_triple_batch(
            triple_list1, triple_list2, triple_set1, triple_set2,
            entity_list1, entity_list2, batch_size, step, neighbor1, neighbor2, neg_triples_num)
        batches.append((pos, neg))
    return batches


def generate_neighbours(entity_embeds, entity_list, neighbors_num, threads_num):
    entity_arr = np.array(entity_list)
    ent_frags = task_divide(entity_arr, threads_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list))), threads_num)

    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = []
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(
            _find_neighbours,
            args=(ent_frags[i], entity_arr, entity_embeds[ent_frag_indexes[i], :], entity_embeds, neighbors_num)))
    pool.close()
    pool.join()

    dic = {}
    for res in results:
        dic.update(res.get())
    del results
    gc.collect()
    return dic


def _find_neighbours(frags, entity_list, sub_embed, embed, k):
    dic = {}
    sim_mat = np.matmul(sub_embed, embed.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        dic[frags[i]] = entity_list[sort_index[:k]].tolist()
    return dic
