"""MLP model for BERT-INT interaction scoring."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class InteractionMLP(nn.Module):
    """Multi-layer perceptron for scoring entity pair interactions.

    Architecture:
        Input (85 features: 42 neighbor + 42 attribute + 1 description)
        → Linear(input_dim → hidden_dim) + ReLU
        → Linear(hidden_dim → 1) + Tanh
        → Output (score ∈ [-1, 1])
    """

    def __init__(self, input_dim: int = 85, hidden_dim: int = 11):
        """Initialize interaction MLP.

        Args:
            input_dim: Input feature dimension (default: 85)
                - 42 from neighbor-view (21 kernels × 2 pooling methods)
                - 42 from attribute-view (21 kernels × 2 pooling methods)
                - 1 from description-view (cosine similarity)
            hidden_dim: Hidden layer dimension (default: 11)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP
        self.dense1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.dense2 = nn.Linear(hidden_dim, 1, bias=True)

        # Xavier initialization
        init.xavier_normal_(self.dense1.weight)
        init.xavier_normal_(self.dense2.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            features: Input features of shape [batch_size, input_dim]

        Returns:
            Scores of shape [batch_size] with values in [-1, 1]
        """
        x = self.dense1(features)  # [B, hidden_dim]
        x = F.relu(x)
        x = self.dense2(x)  # [B, 1]
        x = F.tanh(x)
        x = torch.squeeze(x, dim=1)  # [B]
        return x

    def __repr__(self) -> str:
        return (
            f"InteractionMLP(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )
