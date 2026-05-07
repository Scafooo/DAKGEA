"""PyTorch loss functions for MultiKE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def relation_logistic_loss(phs, prs, pts, nhs, nrs, nts):
    pos_score = -torch.sum((phs + prs - pts) ** 2, dim=1)
    neg_score = -torch.sum((nhs + nrs - nts) ** 2, dim=1)
    return torch.sum(F.softplus(-pos_score)) + torch.sum(F.softplus(neg_score))


def relation_logistic_loss_wo_negs(phs, prs, pts):
    pos_score = -torch.sum((phs + prs - pts) ** 2, dim=1)
    return torch.sum(F.softplus(-pos_score))


def logistic_loss_wo_negs(phs, pas, pvs, pws):
    pos_score = -torch.sum((phs + pas - pvs) ** 2, dim=1)
    return torch.sum(F.softplus(-pos_score) * pws)


def space_mapping_loss(view_embeds, shared_embeds, mapping, eye_mat, orthogonal_weight, norm_w=0.0001):
    mapped = F.normalize(view_embeds @ mapping, dim=1)
    map_loss = torch.sum((shared_embeds - mapped) ** 2)
    norm_loss = torch.sum(mapping ** 2)
    orth_loss = torch.sum((mapping @ mapping.t() - eye_mat) ** 2)
    return map_loss + orthogonal_weight * orth_loss + norm_w * norm_loss


def alignment_loss(ents1, ents2):
    return torch.sum((ents1 - ents2) ** 2)
