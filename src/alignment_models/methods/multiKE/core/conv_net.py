"""AttributeConvNet: PyTorch Module replacing the conv() function in MultiKE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeConvNet(nn.Module):
    """Scores attribute triples (entity, attribute, value) via CNN."""

    def __init__(self, dim: int, feature_map_size: int = 2,
                 kernel_size=(2, 4), layer_num: int = 2):
        super().__init__()
        self.dim = dim
        self.batch_norm = nn.BatchNorm2d(1)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=feature_map_size if i > 0 else 1,
                out_channels=feature_map_size,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            )
            for i in range(layer_num)
        ])
        # Compute flat dim: after conv over [B, feature_map_size, 2, dim]
        # with "same" padding (approx): output = [B, feature_map_size, 2, dim]
        flat_dim = feature_map_size * 2 * dim
        self.dense = nn.Linear(flat_dim, dim)
        self._flat_dim = flat_dim

    def forward(self, attr_hs, attr_as, attr_vs):
        """
        attr_hs: [B, dim] entity embeddings
        attr_as: [B, dim] attribute embeddings
        attr_vs: [B, dim] value embeddings
        Returns: [B] scores (higher = more plausible)
        """
        B = attr_hs.shape[0]
        avs = torch.cat([attr_as.unsqueeze(1), attr_vs.unsqueeze(1)], dim=1)  # [B, 2, dim]
        x = avs.unsqueeze(1)  # [B, 1, 2, dim]

        x = self.batch_norm(x)
        for conv in self.conv_layers:
            x = torch.tanh(conv(x))
        x = F.normalize(x, dim=2)

        flat = x.reshape(B, -1)
        # Pad or trim to self._flat_dim if shape mismatch (due to padding)
        if flat.shape[1] != self._flat_dim:
            if flat.shape[1] > self._flat_dim:
                flat = flat[:, :self._flat_dim]
            else:
                pad = torch.zeros(B, self._flat_dim - flat.shape[1], device=flat.device)
                flat = torch.cat([flat, pad], dim=1)

        out = torch.tanh(self.dense(flat))
        out = F.normalize(out, dim=1)

        score = -torch.sum((attr_hs - out) ** 2, dim=1)
        return score
