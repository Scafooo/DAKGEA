"""Predicate alignment for MultiKE: Levenshtein + embedding similarity."""

from __future__ import annotations

import numpy as np
from sklearn import preprocessing

try:
    import Levenshtein as _lev
    _levenshtein_ratio = _lev.ratio
except ImportError:
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = tmp
        return 1.0 - dp[n] / max(m, n)


def _get_predicate_match_dict(p_ln_dict_1, p_ln_dict_2):
    match_dict, sim_dict = {}, {}
    for p1, ln1 in p_ln_dict_1.items():
        match_p2 = ''
        max_sim = 0.0
        for p2, ln2 in p_ln_dict_2.items():
            sim = _levenshtein_ratio(ln1, ln2)
            if sim > max_sim:
                match_p2 = p2
                max_sim = sim
        match_dict[p1] = match_p2
        sim_dict[p1] = max_sim
    return match_dict, sim_dict


def init_predicate_alignment(p_ln_dict_1, p_ln_dict_2, predicate_init_sim):
    match_1_2, sim_1 = _get_predicate_match_dict(p_ln_dict_1, p_ln_dict_2)
    match_2_1, sim_2 = _get_predicate_match_dict(p_ln_dict_2, p_ln_dict_1)

    match_pairs_set = set()
    latent_sim_dict = {}
    for p1, p2 in match_1_2.items():
        if match_2_1.get(p2) == p1:
            latent_sim_dict[(p1, p2)] = sim_1[p1]
            if sim_1[p1] > predicate_init_sim:
                match_pairs_set.add((p1, p2, sim_1[p1]))
    return match_pairs_set, latent_sim_dict


def predicate2id_matched_pairs(predicate_match_pairs_set, p_id_dict_1, p_id_dict_2):
    id_match = set()
    for p1, p2, w in predicate_match_pairs_set:
        if p1 in p_id_dict_1 and p2 in p_id_dict_2:
            id_match.add((p_id_dict_1[p1], p_id_dict_2[p2], w))
    return id_match


def _link2dic(links):
    dic1, dic2 = {}, {}
    for i, j, w in links:
        dic1[i] = (j, w)
        dic2[j] = (i, w)
    return dic1, dic2


def generate_sup_predicate_triples(predicate_links, triples1, triples2):
    link_dic1, link_dic2 = _link2dic(predicate_links)
    sup1, sup2 = set(), set()
    for s, p, o in triples1:
        if p in link_dic1:
            sup1.add((s, link_dic1[p][0], o, link_dic1[p][1]))
    for s, p, o in triples2:
        if p in link_dic2:
            sup2.add((s, link_dic2[p][0], o, link_dic2[p][1]))
    return list(sup1), list(sup2)


def _zoom_weight(weight, min_w_before, min_w_after=0.5):
    return 1.0 - (1.0 - weight) * (1.0 - min_w_after) / (1.0 - min_w_before)


def add_weights(predicate_links, triples1, triples2, min_w_before):
    link_dic1, link_dic2 = _link2dic(predicate_links)
    w_default = 0.2
    wt1, wt2 = set(), set()
    for s, p, o in triples1:
        if p in link_dic1:
            wt1.add((s, p, o, _zoom_weight(link_dic1[p][1], min_w_before)))
        else:
            wt1.add((s, p, o, w_default))
    for s, p, o in triples2:
        if p in link_dic2:
            wt2.add((s, p, o, _zoom_weight(link_dic2[p][1], min_w_before)))
        else:
            wt2.add((s, p, o, w_default))
    return list(wt1), list(wt2), wt1, wt2


def find_predicate_alignment_by_embedding(embed, predicate_list1, predicate_list2,
                                           predicate_id_dict1, predicate_id_dict2):
    embed = preprocessing.normalize(embed)
    sim_mat = np.matmul(embed, embed.T)

    matched_1, matched_2 = {}, {}
    for i in predicate_list1:
        for j in (-sim_mat[i, :]).argsort():
            if j in predicate_list2:
                matched_1[i] = j
                break
    for j in predicate_list2:
        for i in (-sim_mat[j, :]).argsort():
            if i in predicate_list1:
                matched_2[j] = i
                break

    latent_sim = {}
    for i, j in matched_1.items():
        if matched_2.get(j) == i:
            latent_sim[(i, j)] = sim_mat[i, j]
    return latent_sim


def extract_local_name(uri: str) -> str:
    if '#' in uri:
        return uri.split('#')[-1].replace('_', ' ')
    return uri.split('/')[-1].replace('_', ' ')


class PredicateAlignModel:
    def __init__(self, kgs, predicate_init_sim: float = 0.90, predicate_soft_sim: float = 0.85,
                 relation_name_dict1=None, relation_name_dict2=None,
                 attribute_name_dict1=None, attribute_name_dict2=None):
        self.kgs = kgs
        self.predicate_init_sim = predicate_init_sim
        self.predicate_soft_sim = predicate_soft_sim

        # Build local name dicts from KG predicates if not provided
        self.relation_name_dict1 = relation_name_dict1 or {
            p: extract_local_name(p) for p in kgs.kg1.entities_id_dict  # populated below
        }
        # Correct: use the original URI KGs for predicate names
        self.relation_name_dict1 = relation_name_dict1 or {
            p: extract_local_name(p) for p in kgs.uri_kg1.relations_set
        }
        self.relation_name_dict2 = relation_name_dict2 or {
            p: extract_local_name(p) for p in kgs.uri_kg2.relations_set
        }
        self.attribute_name_dict1 = attribute_name_dict1 or {
            p: extract_local_name(p) for p in kgs.uri_kg1.attributes_set
        }
        self.attribute_name_dict2 = attribute_name_dict2 or {
            p: extract_local_name(p) for p in kgs.uri_kg2.attributes_set
        }

        self.relation_id_alignment_set = None
        self.sup_relation_alignment_triples1 = self.sup_relation_alignment_triples2 = None
        self.relation_triples_w_weights1 = self.relation_triples_w_weights2 = None
        self.relation_triples_w_weights_set1 = self.relation_triples_w_weights_set2 = None

        self.attribute_id_alignment_set = None
        self.sup_attribute_alignment_triples1 = self.sup_attribute_alignment_triples2 = None
        self.attribute_triples_w_weights1 = self.attribute_triples_w_weights2 = None
        self.attribute_triples_w_weights_set1 = self.attribute_triples_w_weights_set2 = None

        self.relation_alignment_set, self.relation_latent_sim_init = init_predicate_alignment(
            self.relation_name_dict1, self.relation_name_dict2, predicate_init_sim)
        self.attribute_alignment_set, self.attribute_latent_sim_init = init_predicate_alignment(
            self.attribute_name_dict1, self.attribute_name_dict2, predicate_init_sim)
        self.relation_alignment_set_init = self.relation_alignment_set
        self.attribute_alignment_set_init = self.attribute_alignment_set

        self.update_relation_triples(self.relation_alignment_set)
        self.update_attribute_triples(self.attribute_alignment_set)

    def update_attribute_triples(self, attribute_alignment_set):
        self.attribute_id_alignment_set = predicate2id_matched_pairs(
            attribute_alignment_set, self.kgs.kg1.attributes_id_dict, self.kgs.kg2.attributes_id_dict)
        self.sup_attribute_alignment_triples1, self.sup_attribute_alignment_triples2 = \
            generate_sup_predicate_triples(self.attribute_id_alignment_set,
                                           self.kgs.kg1.local_attribute_triples_list,
                                           self.kgs.kg2.local_attribute_triples_list)
        (self.attribute_triples_w_weights1, self.attribute_triples_w_weights2,
         self.attribute_triples_w_weights_set1, self.attribute_triples_w_weights_set2) = \
            add_weights(self.attribute_id_alignment_set,
                        self.kgs.kg1.local_attribute_triples_list,
                        self.kgs.kg2.local_attribute_triples_list,
                        self.predicate_soft_sim)

    def update_relation_triples(self, relation_alignment_set):
        self.relation_id_alignment_set = predicate2id_matched_pairs(
            relation_alignment_set, self.kgs.kg1.relations_id_dict, self.kgs.kg2.relations_id_dict)
        self.sup_relation_alignment_triples1, self.sup_relation_alignment_triples2 = \
            generate_sup_predicate_triples(self.relation_id_alignment_set,
                                           self.kgs.kg1.local_relation_triples_list,
                                           self.kgs.kg2.local_relation_triples_list)
        (self.relation_triples_w_weights1, self.relation_triples_w_weights2,
         self.relation_triples_w_weights_set1, self.relation_triples_w_weights_set2) = \
            add_weights(self.relation_id_alignment_set,
                        self.kgs.kg1.local_relation_triples_list,
                        self.kgs.kg2.local_relation_triples_list,
                        self.predicate_soft_sim)

    def update_predicate_alignment(self, embed, predicate_type: str = 'relation', w: float = 0.7):
        if predicate_type == 'relation':
            p_list1 = self.kgs.kg1.relations_list
            p_list2 = self.kgs.kg2.relations_list
            p_id_dict1 = self.kgs.kg1.relations_id_dict
            p_id_dict2 = self.kgs.kg2.relations_id_dict
            alignment_set_init = self.relation_alignment_set_init
        else:
            p_list1 = self.kgs.kg1.attributes_list
            p_list2 = self.kgs.kg2.attributes_list
            p_id_dict1 = self.kgs.kg1.attributes_id_dict
            p_id_dict2 = self.kgs.kg2.attributes_id_dict
            alignment_set_init = self.attribute_alignment_set_init

        latent_sim = find_predicate_alignment_by_embedding(
            embed, p_list1, p_list2, p_id_dict1, p_id_dict2)

        new_alignment = set()
        for (p1, p2, sim_init) in alignment_set_init:
            p_id_1 = p_id_dict1[p1]
            p_id_2 = p_id_dict2[p2]
            sim = sim_init
            if (p_id_1, p_id_2) in latent_sim:
                sim = w * sim + (1 - w) * latent_sim[(p_id_1, p_id_2)]
            if sim > self.predicate_soft_sim:
                new_alignment.add((p1, p2, sim))

        if predicate_type == 'relation':
            self.relation_alignment_set = new_alignment
            self.update_relation_triples(new_alignment)
        else:
            self.attribute_alignment_set = new_alignment
            self.update_attribute_triples(new_alignment)
