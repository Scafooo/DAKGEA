"""ID generation and triple/link ID mapping utilities for MultiKE."""

from __future__ import annotations

import numpy as np


def sort_elements(triples, elements_set):
    dic = {}
    for s, p, o in triples:
        for e in (s, p, o):
            if e in elements_set:
                dic[e] = dic.get(e, 0) + 1
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return [x[0] for x in sorted_list], dic


def generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = {}, {}
    if ordered:
        kg1_ordered, _ = sort_elements(kg1_triples, kg1_elements)
        kg2_ordered, _ = sort_elements(kg2_triples, kg2_elements)
        n1, n2 = len(kg1_ordered), len(kg2_ordered)
        n = max(n1, n2)
        for i in range(n):
            if i < n1 and i < n2:
                ids1[kg1_ordered[i]] = i * 2
                ids2[kg2_ordered[i]] = i * 2 + 1
            elif i >= n1:
                ids2[kg2_ordered[i]] = n1 * 2 + (i - n1)
            else:
                ids1[kg1_ordered[i]] = n2 * 2 + (i - n2)
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_sharing_id(train_links, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = {}, {}
    if ordered:
        linked_dic = {y: x for x, y in train_links}
        kg2_linked = [x[1] for x in train_links]
        kg2_unlinked = set(kg2_elements) - set(kg2_linked)
        ids1, ids2 = generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_unlinked, ordered=ordered)
        for ele in kg2_linked:
            ids2[ele] = ids1[linked_dic[ele]]
    else:
        index = 0
        for e1, e2 in train_links:
            ids1[e1] = index
            ids2[e2] = index
            index += 1
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def uris_pair_2ids(uris, ids1, ids2):
    return [(ids1[u1], ids2[u2]) for u1, u2 in uris]


def uris_relation_triple_2ids(uris, ent_ids, rel_ids):
    result = []
    for u1, u2, u3 in uris:
        result.append((ent_ids[u1], rel_ids[u2], ent_ids[u3]))
    return result


def uris_attribute_triple_2ids(uris, ent_ids, attr_ids):
    result = []
    for u1, u2, u3 in uris:
        if u1 in ent_ids and u2 in attr_ids:
            result.append((ent_ids[u1], attr_ids[u2], u3))
    return result


def generate_sup_relation_triples(sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
    new1, new2 = set(), set()
    for e1, e2 in sup_links:
        for r, t in rt_dict1.get(e1, set()):
            new1.add((e2, r, t))
        for h, r in hr_dict1.get(e1, set()):
            new1.add((h, r, e2))
        for r, t in rt_dict2.get(e2, set()):
            new2.add((e1, r, t))
        for h, r in hr_dict2.get(e2, set()):
            new2.add((h, r, e1))
    return new1, new2


def generate_sup_attribute_triples(sup_links, av_dict1, av_dict2):
    new1, new2 = set(), set()
    for e1, e2 in sup_links:
        for a, v in av_dict1.get(e1, set()):
            new1.add((e2, a, v))
        for a, v in av_dict2.get(e2, set()):
            new2.add((e1, a, v))
    return new1, new2
