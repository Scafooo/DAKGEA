import torch
import torch.nn.functional as F
import numpy as np

from ..tools.Announce import Announce


def cos_sim_mat_generate(emb1, emb2, device, bs=128):
    emb1 = emb1.to(device) if hasattr(emb1, 'to') else torch.FloatTensor(emb1).to(device)
    emb2 = emb2.to(device) if hasattr(emb2, 'to') else torch.FloatTensor(emb2).to(device)
    array_emb1 = F.normalize(emb1.float(), p=2, dim=1)
    array_emb2 = F.normalize(emb2.float(), p=2, dim=1)
    res_mat = batch_mat_mm(array_emb1, array_emb2.t(), device, bs=bs)
    return res_mat


def batch_mat_mm(mat1, mat2, device, bs=128):
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0, axis_0, bs):
        temp_div_mat_1 = mat1[i:min(i + bs, axis_0)]
        res = temp_div_mat_1.mm(mat2)
        res_mat.append(res)
    res_mat = torch.cat(res_mat, 0)
    return res_mat


def batch_topk(mat, bs=128, topn=50, largest=False):
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    topn = min(topn, mat.shape[1])
    if topn == 0 or axis_0 == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    for i in range(0, axis_0, bs):
        temp_div_mat = mat[i:min(i + bs, axis_0)]
        score_mat, index_mat = temp_div_mat.topk(topn, largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score, 0)
    res_index = torch.cat(res_index, 0)
    return res_score, res_index


hits_list = [1, 5, 10, 50]


def _compute_hits(index_mat, correct_indices):
    """Core hits/MRR computation. correct_indices[i] is the gold column index for row i."""
    ent1_num, cands_num = index_mat.shape
    print(Announce.printMessage(), 'index_mat.shape', index_mat.shape)
    result_mat = [
        [1 if index_mat[i][j] == correct_indices[i] else 0 for j in range(cands_num)]
        for i in range(ent1_num)
    ]
    mrr_mat = [
        sum([1 / (j + 1) if index_mat[i][j] == correct_indices[i] else 0 for j in range(cands_num)])
        for i in range(ent1_num)
    ]
    result_title_str = ""
    result_str = ""
    hit_values = []
    for hits_num in hits_list:
        if cands_num < hits_num:
            hit_values.append(0.0)
            continue
        total_hit = sum([sum(ent_list[:hits_num]) for ent_list in result_mat])
        hit_value = total_hit / ent1_num
        result_title_str += ''.join(('Hits@', str(hits_num), '\t'))
        result_str += ''.join((str(hit_value), '\t'))
        hit_values.append(hit_value)
    mrr_value = sum(mrr_mat) / len(mrr_mat) if mrr_mat else 0.0
    result_title_str += 'MRR'
    result_str += str(mrr_value)
    print(result_title_str.strip())
    print(result_str.strip())
    return hit_values, mrr_value


def hits(index_mat):
    """Closed evaluation: entity i in KG1 should match entity i in KG2."""
    correct_indices = list(range(index_mat.shape[0]))
    return _compute_hits(index_mat, correct_indices)


def hits_open(index_mat, correct_indices):
    """Open evaluation: rank each KG1 test entity against ALL KG2 entities.

    correct_indices[i] = position of the gold partner in the full KG2 entity list.
    """
    return _compute_hits(index_mat, correct_indices)
