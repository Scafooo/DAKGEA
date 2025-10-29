"""Dual aggregation utilities used by the interaction model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kernel_mus(n_kernels: int) -> torch.FloatTensor:
    mus = [1.0]
    if n_kernels == 1:
        return torch.FloatTensor(mus)
    bin_size = 1.0 / (n_kernels - 1)
    mus.append(1 - bin_size / 2)
    for _ in range(1, n_kernels - 1):
        mus.append(mus[-1] - bin_size)
    return torch.FloatTensor(mus)


def kernel_sigmas(n_kernels: int) -> torch.FloatTensor:
    sigmas = [0.001]
    if n_kernels == 1:
        return torch.FloatTensor(sigmas)
    sigmas += [0.1] * (n_kernels - 1)
    return torch.FloatTensor(sigmas)


def dual_aggregation_features(
    sim_matrix: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    attn_ne1: torch.Tensor,
    attn_ne2: torch.Tensor,
) -> torch.Tensor:
    """Compute dual-aggregation features as in the original implementation."""

    sim_pool_1, _ = sim_matrix.topk(k=1, dim=-1)
    pooling_value_1 = torch.exp(-((sim_pool_1 - mus) ** 2) / (sigmas ** 2) / 2)
    log_pooling_sum_1 = torch.log(torch.clamp(pooling_value_1, min=1e-10)) * attn_ne1 * 0.01
    log_pooling_sum_1 = torch.sum(log_pooling_sum_1, 1)

    sim_pool_2, _ = torch.transpose(sim_matrix, 1, 2).topk(k=1, dim=-1)
    pooling_value_2 = torch.exp(-((sim_pool_2 - mus) ** 2) / (sigmas ** 2) / 2)
    log_pooling_sum_2 = torch.log(torch.clamp(pooling_value_2, min=1e-10)) * attn_ne2 * 0.01
    log_pooling_sum_2 = torch.sum(log_pooling_sum_2, 1)

    attn_ne2_sum = torch.clamp(attn_ne2.sum(dim=1), min=1e-10)
    attn_ne1_sum = torch.clamp(attn_ne1.sum(dim=1), min=1e-10)
    log_pooling_sum_2 = log_pooling_sum_2 * (1 / attn_ne2_sum)
    log_pooling_sum_1 = log_pooling_sum_1 * (1 / attn_ne1_sum)
    return torch.cat([log_pooling_sum_1, log_pooling_sum_2], dim=-1)
