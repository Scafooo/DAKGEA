"""Dual aggregation kernels used by the interaction features."""

from __future__ import annotations

import torch


def kernel_mus(n_kernels: int) -> torch.FloatTensor:
    if n_kernels == 1:
        return torch.FloatTensor([1.0])
    bin_size = 1.0 / (n_kernels - 1)
    values = [1.0, 1.0 - bin_size / 2]
    for _ in range(1, n_kernels - 1):
        values.append(values[-1] - bin_size)
    return torch.FloatTensor(values)


def kernel_sigmas(n_kernels: int) -> torch.FloatTensor:
    if n_kernels == 1:
        return torch.FloatTensor([0.001])
    return torch.FloatTensor([0.001] + [0.1] * (n_kernels - 1))


def dual_aggregation_features(
    sim_matrix: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    attn_ne1: torch.Tensor,
    attn_ne2: torch.Tensor,
) -> torch.Tensor:
    sim_max_1, _ = sim_matrix.topk(k=1, dim=-1)
    pooling_1 = torch.exp(-((sim_max_1 - mus) ** 2) / (sigmas ** 2) / 2)
    pooled_1 = torch.log(torch.clamp(pooling_1, min=1e-10)) * attn_ne1 * 0.01
    pooled_1 = pooled_1.sum(dim=1)

    sim_max_2, _ = sim_matrix.transpose(1, 2).topk(k=1, dim=-1)
    pooling_2 = torch.exp(-((sim_max_2 - mus) ** 2) / (sigmas ** 2) / 2)
    pooled_2 = torch.log(torch.clamp(pooling_2, min=1e-10)) * attn_ne2 * 0.01
    pooled_2 = pooled_2.sum(dim=1)

    denom_2 = torch.clamp(attn_ne2.sum(dim=1), min=1e-10)
    pooled_2 = pooled_2 * (1.0 / denom_2)

    denom_1 = torch.clamp(attn_ne1.sum(dim=1), min=1e-10)
    pooled_1 = pooled_1 * (1.0 / denom_1)

    return torch.cat([pooled_1, pooled_2], dim=-1)
