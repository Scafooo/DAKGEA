"""Basic BERT unit used in the BERT-INT pipeline."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class BasicBertUnitModel(nn.Module):
    """Wrapper around a transformer encoder that produces entity embeddings."""

    def __init__(
        self,
        encoder_name: str,
        input_dim: int,
        output_dim: int,
        *,
        load_strategy: str = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(encoder_name, cache_dir=cache_dir)
        encoder = self._load_encoder(encoder_name, config, load_strategy, cache_dir)
        self.encoder = encoder
        self.dropout = nn.Dropout(p=0.1)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_vec = sequence_output[:, 0]
        cls_vec = self.dropout(cls_vec)
        return self.output_layer(cls_vec)

    @staticmethod
    def _load_encoder(
        encoder_name: str,
        config,
        strategy: str,
        cache_dir: Optional[str],
    ) -> AutoModel:
        """Load the transformer encoder according to the chosen strategy."""
        strategy = (strategy or "auto").lower()
        if strategy == "config":
            return AutoModel.from_config(config)
        if strategy == "pretrained":
            return AutoModel.from_pretrained(encoder_name, cache_dir=cache_dir)

        # auto: prefer pretrained weights, but fall back to config if unavailable
        try:
            return AutoModel.from_pretrained(encoder_name, cache_dir=cache_dir)
        except OSError:
            return AutoModel.from_config(config)
