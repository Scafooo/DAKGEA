"""Configuration for BART latent space interpolation."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class InterpolationConfig:
    """Configuration for BART interpolation behavior.

    Encapsulates all hyperparameters for the interpolation process.
    """

    # Adaptive alpha parameters
    base_alpha: float = 0.35
    alpha_spread: float = 0.25
    min_alpha: float = 0.05
    max_alpha: float = 0.95

    # Conservative alpha for name/title attributes
    conservative_min_alpha: float = 0.10
    conservative_max_alpha: float = 0.30

    # Generation parameters
    max_new_tokens: int = 32
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.2
    num_beams: int = 1
    repetition_penalty: float = 2.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "InterpolationConfig":
        """Create from configuration dictionary."""
        bart_cfg = config.get("bart", {})
        gen_cfg = bart_cfg.get("generation", {})

        return cls(
            base_alpha=float(bart_cfg.get("base_alpha", 0.35)),
            alpha_spread=float(bart_cfg.get("alpha_spread", 0.25)),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            top_k=int(gen_cfg.get("top_k", 50)),
            top_p=float(gen_cfg.get("top_p", 0.95)),
            temperature=float(gen_cfg.get("temperature", 1.2)),
            num_beams=int(gen_cfg.get("num_beams", 1)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 2.0)),
            length_penalty=float(gen_cfg.get("length_penalty", 1.0)),
            no_repeat_ngram_size=int(gen_cfg.get("no_repeat_ngram_size", 4)),
        )

    def get_generation_kwargs(self, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Get kwargs for BART generation."""
        return {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "early_stopping": True,
            "remove_invalid_values": True,
        }

    def is_conservative_predicate(self, predicate: str) -> bool:
        """Check if predicate requires conservative alpha."""
        conservative_keywords = [
            "name", "givenname", "surname", "fullname", "birthname", "title"
        ]
        predicate_lower = (predicate or "").lower()
        return any(kw in predicate_lower for kw in conservative_keywords)

    def get_alpha_bounds(self, predicate: str) -> tuple[float, float]:
        """Get alpha bounds for a predicate."""
        if self.is_conservative_predicate(predicate):
            return self.conservative_min_alpha, self.conservative_max_alpha
        return self.min_alpha, self.max_alpha
