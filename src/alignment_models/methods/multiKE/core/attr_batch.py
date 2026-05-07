"""Attribute triple batch generation for MultiKE attribute view training."""

from __future__ import annotations

import random


def generate_pos_triples(triples, batch_size, step):
    start = step * batch_size
    end = min(start + batch_size, len(triples))
    return triples[start:end]


def generate_neg_attribute_triples(pos_batch, all_triples_set, entity_list, neg_triples_num, neighbor=None):
    if neighbor is None:
        neighbor = {}
    neg_batch = []
    for head, attribute, value, w in pos_batch:
        for _ in range(neg_triples_num):
            while True:
                neg_head = random.choice(neighbor.get(head, entity_list))
                if (neg_head, attribute, value, w) not in all_triples_set:
                    break
            neg_batch.append((neg_head, attribute, value, w))
    return neg_batch


def generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                    entity_list1, entity_list2, batch_size,
                                    step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos2 = generate_pos_triples(triple_list2, batch_size2, step)
    neg1 = generate_neg_attribute_triples(pos1, triple_set1, entity_list1, neg_triples_num, neighbor=neighbor1)
    neg2 = generate_neg_attribute_triples(pos2, triple_set2, entity_list2, neg_triples_num, neighbor=neighbor2)
    return pos1 + pos2, neg1 + neg2


def generate_all_attribute_batches(triple_list1, triple_list2, triple_set1, triple_set2,
                                   entity_list1, entity_list2, batch_size, triple_steps,
                                   neighbor1, neighbor2):
    batches = []
    for step in range(triple_steps):
        pos, neg = generate_attribute_triple_batch(
            triple_list1, triple_list2, triple_set1, triple_set2,
            entity_list1, entity_list2, batch_size, step, neighbor1, neighbor2, 0)
        batches.append((pos, neg))
    return batches
