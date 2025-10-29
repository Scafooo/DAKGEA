import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from src.alignment_models.methods.hybea import runtime as cfg

log = cfg.logger

def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1)


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.
    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.
    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat.T - nearest_values1
    csls_sim_mat = csls_sim_mat.T - nearest_values2
    return csls_sim_mat


def cos_sim_mat_generate(emb1,emb2,bs = 128,cuda_num = 0, csls=cfg.CSLS):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2,dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2,dim=1)
    res_mat = batch_mat_mm(array_emb1,array_emb2.t(),cuda_num,bs=bs)
    if csls > 0:
        res_mat = csls_sim(res_mat, csls)
    return res_mat



def batch_mat_mm(mat1,mat2,cuda_num,bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].cuda(cuda_num)
        res = temp_div_mat_1.mm(mat2.cuda(cuda_num))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat


# ... existing code ...

def batch_topk(res_mat, batch_size=128, topn=50, largest=True, cuda_num=0):
    # Ensure k does not exceed the last dimension of the matrix for the current batch
    # temp_div_mat is expected to be a 2D tensor [batch, candidates]
    temp_div_mat = res_mat  # assuming res_mat is already the sliced/batch tensor
    if temp_div_mat.dim() < 1:
        raise ValueError("batch_topk received a tensor with invalid dimensions")

    last_dim = temp_div_mat.size(-1)
    if last_dim == 0:
        # Return empty tensors with correct batch size to avoid downstream shape errors
        batch_len = temp_div_mat.size(0) if temp_div_mat.dim() > 0 else 0
        empty_scores = temp_div_mat.new_empty((batch_len, 0))
        empty_indices = temp_div_mat.new_empty((batch_len, 0), dtype=torch.long)
        return empty_scores, empty_indices

    k = topn if topn <= last_dim else last_dim
    score_mat, index_mat = temp_div_mat.topk(k, largest=largest)
    return score_mat, index_mat

# ... existing code ...

# def batch_topk(mat,bs=128,topn = 50,largest = False,cuda_num = 0):
#     #be equal to topk, Speed up computing with GPU
#     res_score = []
#     res_index = []
#     axis_0 = mat.shape[0]
#     for i in range(0,axis_0,bs):
#         temp_div_mat = mat[i:min(i+bs,axis_0)].cuda(cuda_num)
#         score_mat,index_mat =temp_div_mat.topk(topn,largest=largest)
#         res_score.append(score_mat.cpu())
#         res_index.append(index_mat.cpu())
#     res_score = torch.cat(res_score,0)
#     res_index = torch.cat(res_index,0)
#     return res_score,res_index


def hit_res(index_mat):
    
    ent1_num,ent2_num = index_mat.shape
    topk_n = [0 for _ in range(ent2_num)]
    rank_MR = 0
    rank_MRR = 0
    entities_number = 0
    for i in range(ent1_num):
        for j in range(ent2_num):
            if index_mat[i][j].item() == i:
                rank_MR += (j + 1)
                rank_MRR += 1 / (j + 1)
                entities_number += 1
                for h in range(j,ent2_num):
                    topk_n[h]+=1
                break
    topk_n = [round(x/ent1_num,5) for x in topk_n]
    def _hit_at(index: int) -> float:
        if ent2_num == 0:
            return 0.0
        idx = min(index, ent2_num - 1)
        return topk_n[idx] * 100

    hits1 = _hit_at(0)
    hits5 = _hit_at(4)
    hits10 = _hit_at(9)
    hits25 = _hit_at(24) if ent2_num >= 25 else None
    hits50 = _hit_at(49) if ent2_num >= 50 else None

    log.debug(
        "Raw ranking stats: MR=%s MRR=%s ent1=%d ent2=%d",
        rank_MR,
        rank_MRR,
        ent1_num,
        ent2_num,
    )

    rank_MR /= entities_number
    rank_MRR /= entities_number
    info_parts = [
        f"Hits@1 {hits1:.2f}%",
        f"Hits@5 {hits5:.2f}%",
        f"Hits@10 {hits10:.2f}%",
    ]
    if hits25 is not None:
        info_parts.append(f"Hits@25 {hits25:.2f}%")
    if hits50 is not None:
        info_parts.append(f"Hits@50 {hits50:.2f}%")
    info_parts.append(f"MR {rank_MR:.4f}")
    info_parts.append(f"MRR {rank_MRR:.4f}")
    info_parts.append(f"entities {entities_number}")
    log.info(" | ".join(info_parts))
    
    return topk_n[1 - 1]*100
