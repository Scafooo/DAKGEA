"""BART latent space interpolator for attribute value generation.

Extracted from bart_interpolator.py, focusing only on interpolation logic.
"""

import re
import torch
from typing import Tuple, Optional
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ..models import InterpolationConfig


def _simple_clean(x: str) -> str:
    """Simple text cleaning function."""
    if not x:
        return x
    x = re.sub(r"http\S+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s\.\-']", " ", x)
    x = re.sub(r"\b(\w+)( \1\b)+", r"\1", x, flags=re.IGNORECASE)
    x = re.sub(r"\b([A-Za-z])\b", "", x)
    return re.sub(r"\s+", " ", x).strip(" .-")


class BARTInterpolator:
    """Handles BART latent space interpolation for generating attribute values.

    This class encapsulates the interpolation logic that was previously embedded
    in BartInterpolatorPLM, making it cleaner and more focused.
    """

    def __init__(
        self,
        model: BartForConditionalGeneration,
        tokenizer: BartTokenizer,
        config: InterpolationConfig,
        device: str = "cpu",
        max_len_in: int = 96,
    ):
        """Initialize the interpolator.

        Args:
            model: Fine-tuned BART model
            tokenizer: BART tokenizer
            config: Interpolation configuration
            device: Device to run on ('cpu' or 'cuda')
            max_len_in: Maximum input sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.max_len_in = max_len_in

    def interpolate_pair(
        self,
        val_src: str,
        val_tgt: str,
        predicate: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Interpolate between source and target values using latent mixing.

        Args:
            val_src: Source value
            val_tgt: Target value
            predicate: Predicate name (for conservative alpha on names)
            max_new_tokens: Maximum tokens to generate (overrides config)

        Returns:
            Tuple of (interpolated_src, interpolated_tgt)
        """
        if not val_src and not val_tgt:
            return "", ""
        if not val_src:
            t = _simple_clean(val_tgt)
            return t, t
        if not val_tgt:
            s = _simple_clean(val_src)
            return s, s

        # Tokenize both values
        toks = self.tokenizer(
            [val_src, val_tgt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len_in,
        ).to(self.device)

        # Encode
        self.model.eval()
        with torch.no_grad():
            enc = self.model.get_encoder()(
                toks.input_ids, attention_mask=toks.attention_mask
            )

        # Split hidden states and masks
        h1 = enc.last_hidden_state[0]  # (seq1, dim)
        h2 = enc.last_hidden_state[1]  # (seq2, dim)
        a1 = toks.attention_mask[0]  # (seq1,)
        a2 = toks.attention_mask[1]  # (seq2,)

        # Compute adaptive alpha
        m1 = self._mean_pool(h1, a1)
        m2 = self._mean_pool(h2, a2)
        alpha = self._adaptive_alpha(m1, m2, predicate)

        # Asymmetric latent mixing
        h_mix_src = (1 - alpha) * h1 + alpha * h2
        h_mix_tgt = (1 - alpha) * h2 + alpha * h1

        # Prepare for decoding
        enc_src = BaseModelOutput(last_hidden_state=h_mix_src.unsqueeze(0))
        enc_tgt = BaseModelOutput(last_hidden_state=h_mix_tgt.unsqueeze(0))
        mask_src = a1.unsqueeze(0)
        mask_tgt = a2.unsqueeze(0)

        start = torch.tensor(
            [[self.model.config.decoder_start_token_id]], device=self.device
        )
        bad = self.tokenizer.convert_tokens_to_ids(["<SRC>", "<TGT>", "<SEP>"])

        # Generation kwargs
        gen_kwargs = self.config.get_generation_kwargs(max_new_tokens)
        gen_kwargs["decoder_input_ids"] = start
        gen_kwargs["bad_words_ids"] = [[i] for i in bad]

        # Generate
        with torch.no_grad():
            ids_src = self.model.generate(encoder_outputs=enc_src, **gen_kwargs)
            ids_tgt = self.model.generate(encoder_outputs=enc_tgt, **gen_kwargs)

        # Decode and clean
        out_src = _simple_clean(
            self.tokenizer.decode(ids_src[0], skip_special_tokens=True)
        )
        out_tgt = _simple_clean(
            self.tokenizer.decode(ids_tgt[0], skip_special_tokens=True)
        )

        return out_src, out_tgt

    def _mean_pool(
        self, H: torch.Tensor, attn: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean pooling over sequence dimension.

        Args:
            H: Hidden states (seq, dim) or (1, seq, dim)
            attn: Attention mask (seq,) or (1, seq)

        Returns:
            Pooled representation (dim,)
        """
        if H.dim() == 3:
            H = H.squeeze(0)
        if attn is None:
            return H.mean(0)
        mask = attn.squeeze(0).unsqueeze(-1).float()  # (seq, 1)
        return (H * mask).sum(0) / mask.sum(0).clamp_min(1.0)

    def _adaptive_alpha(
        self, h1_mean: torch.Tensor, h2_mean: torch.Tensor, predicate: str
    ) -> float:
        """Compute adaptive interpolation coefficient based on similarity.

        Args:
            h1_mean: Mean-pooled representation of first value
            h2_mean: Mean-pooled representation of second value
            predicate: Predicate name (for conservative bounds)

        Returns:
            Alpha value in appropriate range
        """
        import torch.nn.functional as F

        # Cosine similarity ∈ [-1, 1] -> remap to [0,1]
        cos = F.cosine_similarity(h1_mean.unsqueeze(0), h2_mean.unsqueeze(0)).item()
        sim01 = (cos + 1.0) / 2.0

        # α = base ± spread * (2*sim - 1)
        alpha = self.config.base_alpha + self.config.alpha_spread * (2 * sim01 - 1)

        # Get bounds (conservative for names/titles)
        min_alpha, max_alpha = self.config.get_alpha_bounds(predicate)

        # Clamp
        return max(min_alpha, min(max_alpha, alpha))
