"""
Advanced training modules for BART fine-tuning.
Each module can be independently enabled/disabled via configuration.
"""

import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PairExample:
    """Training example with metadata for advanced techniques."""
    val_src: str
    val_tgt: str
    predicate: str
    predicate_type: Optional[str] = None  # For multi-task learning
    difficulty: Optional[float] = None  # For curriculum learning


# ============================================================================
# 1. Stratified Sampling
# ============================================================================

class StratifiedSampler:
    """Balance predicate distribution in training data."""

    def __init__(
        self,
        min_samples: int = 10,
        max_samples: int = 1000,
        strategy: str = "sqrt",  # "uniform", "sqrt", "log"
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.strategy = strategy
        logger.info(f"[StratifiedSampler] Initialized with strategy={strategy}, "
                   f"min={min_samples}, max={max_samples}")

    def sample(self, examples_by_predicate: Dict[str, List[PairExample]]) -> List[PairExample]:
        """Apply stratified sampling to balance predicates."""
        sampled = []
        predicate_counts = {p: len(exs) for p, exs in examples_by_predicate.items()}
        total_examples = sum(predicate_counts.values())

        logger.info(f"[StratifiedSampler] Original distribution: {len(examples_by_predicate)} predicates, {total_examples} examples")

        for predicate, examples in examples_by_predicate.items():
            original_count = len(examples)

            if self.strategy == "uniform":
                # All predicates get same number of samples
                target_count = min(self.max_samples, max(self.min_samples, original_count))
            elif self.strategy == "sqrt":
                # Square root scaling: reduces over-representation of frequent predicates
                avg_count = total_examples / len(examples_by_predicate)
                target_count = int(np.sqrt(original_count) * np.sqrt(avg_count))
                target_count = min(self.max_samples, max(self.min_samples, target_count))
            elif self.strategy == "log":
                # Logarithmic scaling: more aggressive balancing
                avg_count = total_examples / len(examples_by_predicate)
                target_count = int(np.log1p(original_count) * avg_count / np.log1p(avg_count))
                target_count = min(self.max_samples, max(self.min_samples, target_count))
            else:
                target_count = original_count

            # Sample or oversample to reach target
            if original_count >= target_count:
                sampled_examples = random.sample(examples, target_count)
            else:
                # Oversample with replacement
                sampled_examples = random.choices(examples, k=target_count)

            sampled.extend(sampled_examples)
            logger.debug(f"  [{predicate}] {original_count} → {target_count} samples")

        random.shuffle(sampled)
        logger.info(f"[StratifiedSampler] Final dataset: {len(sampled)} examples")
        return sampled


# ============================================================================
# 2. Contrastive Learning
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better representations."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,  # (batch_size, hidden_dim)
        positive: torch.Tensor,  # (batch_size, hidden_dim)
        negatives: torch.Tensor,  # (batch_size, num_negatives, hidden_dim)
    ) -> torch.Tensor:
        """
        Compute contrastive loss with one positive and multiple negatives per anchor.
        """
        batch_size = anchor.size(0)
        num_negatives = negatives.size(1)

        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch_size,)

        # Negative similarities
        neg_sim = torch.bmm(
            negatives, anchor.unsqueeze(-1)
        ).squeeze(-1) / self.temperature  # (batch_size, num_negatives)

        # Concatenate positive and negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss


class NegativeSampler:
    """Generate hard negative examples for contrastive learning."""

    def __init__(
        self,
        num_negatives: int = 3,
        strategy: str = "same_predicate",  # "same_predicate", "same_type", "random"
    ):
        self.num_negatives = num_negatives
        self.strategy = strategy
        self.examples_by_predicate: Dict[str, List[PairExample]] = defaultdict(list)
        self.examples_by_type: Dict[str, List[PairExample]] = defaultdict(list)
        self.all_examples: List[PairExample] = []

    def index_examples(self, examples: List[PairExample]):
        """Build index for efficient negative sampling."""
        self.examples_by_predicate.clear()
        self.examples_by_type.clear()
        self.all_examples = examples

        for ex in examples:
            self.examples_by_predicate[ex.predicate].append(ex)
            if ex.predicate_type:
                self.examples_by_type[ex.predicate_type].append(ex)

        logger.info(f"[NegativeSampler] Indexed {len(examples)} examples")

    def sample_negatives(self, anchor: PairExample) -> List[PairExample]:
        """Sample hard negative examples for the given anchor."""
        if self.strategy == "same_predicate":
            # Negatives from same predicate (hardest)
            candidates = [
                ex for ex in self.examples_by_predicate[anchor.predicate]
                if ex.val_src != anchor.val_src or ex.val_tgt != anchor.val_tgt
            ]
        elif self.strategy == "same_type" and anchor.predicate_type:
            # Negatives from same type but different predicate
            candidates = [
                ex for ex in self.examples_by_type[anchor.predicate_type]
                if ex.predicate != anchor.predicate
            ]
        else:
            # Random negatives (easiest)
            candidates = [
                ex for ex in self.all_examples
                if ex.val_src != anchor.val_src or ex.val_tgt != anchor.val_tgt
            ]

        # Sample negatives
        if len(candidates) >= self.num_negatives:
            return random.sample(candidates, self.num_negatives)
        else:
            # Not enough candidates, sample with replacement
            return random.choices(candidates, k=self.num_negatives) if candidates else []


# ============================================================================
# 3. Multi-Task Learning
# ============================================================================

class AttributeTypeClassifier(nn.Module):
    """Auxiliary classifier for predicting attribute types."""

    def __init__(self, hidden_size: int, num_types: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_types),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, num_types)
        """
        # Pool sequence (mean pooling)
        pooled = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled)


class PredicateMatchClassifier(nn.Module):
    """Auxiliary classifier for predicting if two values share a predicate."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),  # Binary: match or not
        )

    def forward(self, hidden1: torch.Tensor, hidden2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden1, hidden2: (batch_size, hidden_size)
        Returns:
            logits: (batch_size, 2)
        """
        combined = torch.cat([hidden1, hidden2], dim=-1)
        return self.classifier(combined)


class AttributeTypeInference:
    """Infer attribute types from predicate names and values."""

    TYPE_KEYWORDS = {
        "name": ["name", "label", "title", "username"],
        "date": ["date", "time", "year", "birth", "death", "founded"],
        "number": ["age", "count", "population", "height", "weight", "score"],
        "location": ["location", "place", "city", "country", "address"],
        "description": ["description", "comment", "bio", "summary", "abstract"],
        "identifier": ["id", "uri", "url", "code", "isbn"],
    }

    @staticmethod
    def infer_type(predicate: str, value: str) -> str:
        """Infer attribute type from predicate and value."""
        predicate_lower = predicate.lower()

        # Check predicate keywords
        for type_name, keywords in AttributeTypeInference.TYPE_KEYWORDS.items():
            if any(kw in predicate_lower for kw in keywords):
                return type_name

        # Check value patterns
        if value.replace("-", "").replace("/", "").replace(":", "").isdigit():
            return "date"
        if value.replace(".", "").replace(",", "").isdigit():
            return "number"
        if len(value.split()) > 10:
            return "description"

        return "other"


# ============================================================================
# 4. Advanced Noising
# ============================================================================

class AdvancedNoiser:
    """Advanced noising strategies for BART training."""

    def __init__(
        self,
        span_corruption_ratio: float = 0.3,
        mean_span_length: int = 3,
        entity_aware_masking: bool = True,
        entity_mask_prob: float = 0.5,
        mask_token: str = "<mask>",
    ):
        self.span_corruption_ratio = span_corruption_ratio
        self.mean_span_length = mean_span_length
        self.entity_aware_masking = entity_aware_masking
        self.entity_mask_prob = entity_mask_prob
        self.mask_token = mask_token
        logger.info(f"[AdvancedNoiser] Initialized with span_corruption={span_corruption_ratio}, "
                   f"mean_span_length={mean_span_length}, entity_aware={entity_aware_masking}, "
                   f"entity_mask_prob={entity_mask_prob}")

    def apply_span_corruption(self, text: str) -> str:
        """Apply span corruption: mask contiguous spans of tokens."""
        tokens = text.split()
        if len(tokens) <= 1:
            return text

        num_tokens = len(tokens)
        num_to_mask = max(1, int(num_tokens * self.span_corruption_ratio))

        # Sample span starts
        masked_positions = set()
        while len(masked_positions) < num_to_mask:
            # Sample span start
            start = random.randint(0, num_tokens - 1)
            # Sample span length (Poisson distribution)
            span_length = min(
                np.random.poisson(self.mean_span_length) + 1,
                num_tokens - start
            )
            # Mark positions
            for i in range(start, start + span_length):
                masked_positions.add(i)
                if len(masked_positions) >= num_to_mask:
                    break

        # Apply masking
        masked_tokens = [
            self.mask_token if i in masked_positions else tok
            for i, tok in enumerate(tokens)
        ]

        return " ".join(masked_tokens)

    def detect_entities(self, text: str) -> List[Tuple[int, int]]:
        """Simple heuristic to detect named entities (capitalized sequences)."""
        tokens = text.split()
        entities = []
        start = None

        for i, tok in enumerate(tokens):
            # Check if token is capitalized and contains letters
            if tok and tok[0].isupper() and any(c.isalpha() for c in tok):
                if start is None:
                    start = i
            else:
                if start is not None:
                    entities.append((start, i))
                    start = None

        # Close last entity if needed
        if start is not None:
            entities.append((start, len(tokens)))

        return entities

    def apply_entity_aware_masking(self, text: str) -> str:
        """Mask named entities with higher probability."""
        if not self.entity_aware_masking:
            return self.apply_span_corruption(text)

        tokens = text.split()
        if len(tokens) <= 1:
            return text

        entities = self.detect_entities(text)
        entity_positions = set()
        for start, end in entities:
            for i in range(start, end):
                entity_positions.add(i)

        # Mask entities with higher probability
        masked_tokens = []
        for i, tok in enumerate(tokens):
            if i in entity_positions:
                if random.random() < self.entity_mask_prob:
                    masked_tokens.append(self.mask_token)
                else:
                    masked_tokens.append(tok)
            else:
                # Mask non-entities with lower probability
                if random.random() < self.span_corruption_ratio:
                    masked_tokens.append(self.mask_token)
                else:
                    masked_tokens.append(tok)

        return " ".join(masked_tokens)

    def noise(self, text: str) -> str:
        """Apply noising strategy."""
        if self.entity_aware_masking:
            return self.apply_entity_aware_masking(text)
        else:
            return self.apply_span_corruption(text)


# ============================================================================
# 5. Curriculum Learning
# ============================================================================

class CurriculumScheduler:
    """Schedule curriculum learning phases."""

    def __init__(
        self,
        strategy: str = "length",  # "length", "predicate_frequency", "custom"
        num_phases: int = 3,
        phase_epochs: Optional[List[int]] = None,
    ):
        self.strategy = strategy
        self.num_phases = num_phases
        self.phase_epochs = phase_epochs or [3, 3, 4]
        self.current_phase = 0
        self.current_epoch = 0

    def compute_difficulty(self, example: PairExample) -> float:
        """Compute difficulty score for an example."""
        if self.strategy == "length":
            # Longer values are harder
            total_len = len(example.val_src.split()) + len(example.val_tgt.split())
            return total_len / 20.0  # Normalize
        elif self.strategy == "predicate_frequency":
            # Less frequent predicates are harder (requires external frequency info)
            return example.difficulty if example.difficulty is not None else 0.5
        else:
            # Custom difficulty (should be provided in example)
            return example.difficulty if example.difficulty is not None else 0.5

    def assign_difficulties(self, examples: List[PairExample]) -> List[PairExample]:
        """Assign difficulty scores to examples."""
        for ex in examples:
            if ex.difficulty is None:
                ex.difficulty = self.compute_difficulty(ex)
        return examples

    def get_phase_examples(
        self, all_examples: List[PairExample], phase: int
    ) -> List[PairExample]:
        """Get examples for the current curriculum phase."""
        if phase >= self.num_phases:
            # Final phase: all examples
            return all_examples

        # Sort by difficulty
        sorted_examples = sorted(all_examples, key=lambda ex: ex.difficulty or 0.0)

        # Divide into phases
        phase_size = len(sorted_examples) // self.num_phases
        start_idx = 0
        end_idx = (phase + 1) * phase_size if phase < self.num_phases - 1 else len(sorted_examples)

        # Include all examples from previous phases + current phase
        phase_examples = sorted_examples[:end_idx]

        logger.info(
            f"[Curriculum] Phase {phase + 1}/{self.num_phases}: "
            f"Using {len(phase_examples)}/{len(all_examples)} examples "
            f"(difficulty ≤ {phase_examples[-1].difficulty:.2f})"
        )

        return phase_examples

    def step_epoch(self) -> bool:
        """Step to next epoch. Returns True if phase changed."""
        self.current_epoch += 1
        cumulative_epochs = sum(self.phase_epochs[:self.current_phase + 1])

        if self.current_epoch >= cumulative_epochs and self.current_phase < self.num_phases - 1:
            self.current_phase += 1
            logger.info(f"[Curriculum] Advanced to phase {self.current_phase + 1}/{self.num_phases}")
            return True

        return False


# ============================================================================
# 6. Training Data Augmentation (Placeholder for future implementation)
# ============================================================================

class TrainingAugmenter:
    """Augment training data with synthetic variations."""

    def __init__(
        self,
        synonym_replacement: bool = False,
        back_translation: bool = False,
        random_noise: bool = False,
        augmentation_ratio: float = 0.3,
    ):
        self.synonym_replacement = synonym_replacement
        self.back_translation = back_translation
        self.random_noise = random_noise
        self.augmentation_ratio = augmentation_ratio

    def augment(self, examples: List[PairExample]) -> List[PairExample]:
        """Generate augmented examples with character-level noise.

        Returns:
            Original examples + augmented examples
        """
        if not any([self.synonym_replacement, self.back_translation, self.random_noise]):
            # No augmentation enabled
            return examples

        augmented = []
        num_to_augment = int(len(examples) * self.augmentation_ratio)

        if num_to_augment == 0:
            return examples

        # Sample examples to augment
        import random
        examples_to_augment = random.sample(examples, min(num_to_augment, len(examples)))

        for example in examples_to_augment:
            if self.random_noise:
                # Apply character-level noise
                aug_src = self._add_character_noise(example.val_src)
                aug_tgt = self._add_character_noise(example.val_tgt)

                augmented.append(PairExample(
                    val_src=aug_src,
                    val_tgt=aug_tgt,
                    predicate=example.predicate,
                    predicate_type=example.predicate_type,
                    difficulty=example.difficulty
                ))

        if augmented:
            logger.info(f"[TrainingAugmenter] Generated {len(augmented)} augmented examples with character noise")

        return examples + augmented

    def _add_character_noise(self, text: str, noise_prob: float = 0.1) -> str:
        """Add character-level noise to text.

        Args:
            text: Input text
            noise_prob: Probability of adding noise to each character

        Returns:
            Text with character-level noise
        """
        import random
        import string

        if not text or len(text) < 3:
            return text

        chars = list(text)
        noisy_chars = []

        i = 0
        while i < len(chars):
            char = chars[i]

            # Skip spaces and punctuation
            if char in string.whitespace or char in string.punctuation:
                noisy_chars.append(char)
                i += 1
                continue

            # Decide if we apply noise
            if random.random() < noise_prob:
                noise_type = random.choice(['substitute', 'swap', 'delete', 'insert'])

                if noise_type == 'substitute' and char.isalpha():
                    # Substitute with similar character
                    similar_chars = self._get_similar_chars(char)
                    noisy_chars.append(random.choice(similar_chars) if similar_chars else char)

                elif noise_type == 'swap' and i < len(chars) - 1 and chars[i+1].isalpha():
                    # Swap with next character
                    noisy_chars.append(chars[i+1])
                    noisy_chars.append(char)
                    i += 1  # Skip next char since we already added it

                elif noise_type == 'delete':
                    # Delete character (skip it)
                    pass

                elif noise_type == 'insert':
                    # Insert duplicate or similar character
                    noisy_chars.append(char)
                    if char.isalpha():
                        similar = self._get_similar_chars(char)
                        noisy_chars.append(random.choice(similar) if similar else char)
                    else:
                        noisy_chars.append(char)
                else:
                    # No noise applied
                    noisy_chars.append(char)
            else:
                # No noise
                noisy_chars.append(char)

            i += 1

        return ''.join(noisy_chars)

    def _get_similar_chars(self, char: str) -> List[str]:
        """Get list of similar characters for substitution.

        Args:
            char: Character to find similar ones for

        Returns:
            List of similar characters
        """
        # Common character confusions (visually or phonetically similar)
        similar_map = {
            'a': ['a', 'e', 'o'],
            'e': ['e', 'a', 'i'],
            'i': ['i', 'e', 'y'],
            'o': ['o', 'a', 'u'],
            'u': ['u', 'o', 'v'],
            'b': ['b', 'p', 'd'],
            'p': ['p', 'b'],
            'd': ['d', 'b', 't'],
            't': ['t', 'd'],
            'g': ['g', 'j', 'q'],
            'j': ['j', 'g'],
            'c': ['c', 'k', 's'],
            'k': ['k', 'c'],
            's': ['s', 'c', 'z'],
            'z': ['z', 's'],
            'n': ['n', 'm'],
            'm': ['m', 'n'],
            'w': ['w', 'v'],
            'v': ['v', 'w', 'u'],
            'f': ['f', 'v', 'ph'],
            'l': ['l', 'r'],
            'r': ['r', 'l'],
        }

        char_lower = char.lower()
        similar = similar_map.get(char_lower, [char_lower])

        # Preserve case
        if char.isupper():
            similar = [c.upper() for c in similar]

        return similar
