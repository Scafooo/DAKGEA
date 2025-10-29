"""Implementation of the basic BERT unit encoder used by BERT-INT."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from src.logger import get_logger

logger = get_logger(__name__)


class BasicBertUnitModel(nn.Module):
    """Thin wrapper around HuggingFace BERT to obtain entity embeddings."""

    def __init__(self, encoder_name: str, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        try:
            self.encoder = BertModel.from_pretrained(encoder_name, local_files_only=True)
            logger.debug("[BERT-INT] Loaded pretrained encoder '%s' from local cache.", encoder_name)
        except OSError as exc:
            logger.warning(
                "[BERT-INT] Could not load pretrained encoder '%s' (%s); falling back to random initialisation.",
                encoder_name,
                exc,
            )
            heads = max(1, input_dim // 64)
            while input_dim % heads != 0 and heads > 1:
                heads -= 1
            if input_dim % heads != 0:
                heads = 1
            intermediate = max(input_dim * 4, 256)
            config = BertConfig(
                vocab_size=30522,
                hidden_size=input_dim,
                num_hidden_layers=2,
                num_attention_heads=heads,
                intermediate_size=intermediate,
            )
            self.encoder = BertModel(config)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        sequence_output, _ = outputs[:2]
        cls_repr = sequence_output[:, 0]
        hidden = self.dropout(cls_repr)
        projected = self.output_layer(hidden)
        return projected
