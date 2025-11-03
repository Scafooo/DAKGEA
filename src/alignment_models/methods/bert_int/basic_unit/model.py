"""Neural network module for the BERT-INT basic unit."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.alignment_models.methods.bert_int.config import BasicUnitConfig
from src.logger import get_logger

logger = get_logger(__name__)


class BasicBertUnit(nn.Module):
    """Projection head on top of a HuggingFace BERT encoder."""

    def __init__(self, config: BasicUnitConfig):
        super().__init__()
        self.config = config

        hf_config = self._load_hf_config(config.encoder_name)
        self.encoder = self._load_encoder(config, hf_config)

        hidden_size = hf_config.hidden_size
        expected_size = config.model_input_dim or hidden_size
        if hidden_size != expected_size:
            logger.warning(
                "Basic unit hidden size (%d) differs from configured model_input_dim (%d). Using hidden size.",
                hidden_size,
                expected_size,
            )

        dropout_prob = config.dropout if config.dropout is not None else float(getattr(hf_config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.projection = nn.Linear(hidden_size, config.result_size)

    @staticmethod
    def _load_hf_config(model_name: str):
        try:
            return AutoConfig.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to load AutoConfig for '%s': %s", model_name, exc)
            raise

    @staticmethod
    def _load_encoder(config: BasicUnitConfig, hf_config) -> nn.Module:
        try:
            if config.load_pretrained:
                return AutoModel.from_pretrained(config.encoder_name, config=hf_config)
            return AutoModel.from_config(hf_config)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to initialise encoder '%s': %s", config.encoder_name, exc)
            raise

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a batch of tokenised entities and project to the embedding space."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.projection(self.dropout(cls_embedding))
