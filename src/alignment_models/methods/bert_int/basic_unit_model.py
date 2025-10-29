"""Implementation of the basic BERT unit encoder used by BERT-INT."""

from __future__ import annotations

from transformers import BertModel
import torch
import torch.nn as nn


class BasicBertUnitModel(nn.Module):
    """Thin wrapper around HuggingFace BERT to obtain entity embeddings."""

    def __init__(self, encoder_name: str, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_name)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        sequence_output, _ = outputs[:2]
        cls_repr = sequence_output[:, 0]
        hidden = self.dropout(cls_repr)
        projected = self.output_layer(hidden)
        return projected
