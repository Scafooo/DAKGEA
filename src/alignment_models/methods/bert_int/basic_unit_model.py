"""Torch module implementing the Basic BERT Unit from the original BERT-INT code."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from src.logger import get_logger

logger = get_logger(__name__)


class BasicBertUnitModel(nn.Module):
    """Thin wrapper around HuggingFace's `BertModel` for the Basic BERT Unit step."""

    def __init__(
        self,
        input_size: int,
        result_size: int,
        pretrained_model: str = "bert-base-multilingual-cased",
        dropout: float = 0.1,
        *,
        device: Optional[torch.device] = None,
        load_pretrained: bool = True,
    ) -> None:
        """Initialise the encoder."""
        super().__init__()
        self.result_size = result_size
        self.input_size = input_size
        if load_pretrained:
            try:
                self.bert_model = BertModel.from_pretrained(pretrained_model)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning(
                    "[BERT-INT] Failed to load pretrained model '%s' (%s). Falling back to random init.",
                    pretrained_model,
                    exc,
                )
                self.bert_model = BertModel(BertConfig())
        else:
            self.bert_model = BertModel(BertConfig())
        self.out_linear_layer = nn.Linear(self.input_size, self.result_size)
        self.dropout = nn.Dropout(p=dropout)

        if device is not None:
            self.to(device)

    def forward(self, batch_word_list: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token ids and return projected CLS vectors."""
        outputs = self.bert_model(input_ids=batch_word_list, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        cls_vec = sequence_output[:, 0]
        cls_vec = self.dropout(cls_vec)
        return self.out_linear_layer(cls_vec)
