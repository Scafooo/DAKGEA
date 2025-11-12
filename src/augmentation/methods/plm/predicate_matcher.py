"""
Semantic predicate matching using sentence embeddings.

This module provides sophisticated predicate matching capabilities using
sentence transformer embeddings for semantic similarity.
"""

import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from rdflib import URIRef, Literal

logger = logging.getLogger(__name__)


@dataclass
class PredicateMatch:
    """Represents a match between two predicates."""
    src_predicate: str  # Local name
    tgt_predicate: str  # Local name
    src_uri: URIRef
    tgt_uri: URIRef
    confidence: float  # Similarity score [0, 1]
    strategy: str  # "semantic_embedding"


class PredicateMatcher:
    """
    Semantic predicate matcher using sentence embeddings.

    Uses sentence-transformers to compute semantic similarity between
    predicates, enabling matching even when predicate names differ.

    Features:
    - Semantic embedding-based matching
    - Predicate name expansion (camelCase, snake_case parsing)
    - Embedding cache for performance
    - Configurable similarity threshold
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predicate matcher.

        Args:
            config: Configuration dictionary with keys:
                - embedding_model: Model name (default: "all-MiniLM-L6-v2")
                - similarity_threshold: Min similarity for match (default: 0.7)
                - cache_dir: Directory for embedding cache (default: .cache/embeddings)
                - device: "cuda" or "cpu" (default: auto-detect)
        """
        config = config or {}

        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.cache_dir = Path(config.get("cache_dir", ".cache/embeddings"))
        self.device = config.get("device", None)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model (lazy loading)
        self.model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Load cache from disk if exists
        self._load_cache_from_disk()

        logger.info(f"[PredicateMatcher] Initialized with model={self.embedding_model_name}, "
                   f"threshold={self.similarity_threshold}")

    def _init_model(self):
        """Lazy initialization of the sentence transformer model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"[PredicateMatcher] Loading embedding model: {self.embedding_model_name} on {self.device}")
            self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
            logger.info(f"[PredicateMatcher] Model loaded successfully")

        except ImportError:
            logger.error(
                "[PredicateMatcher] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"[PredicateMatcher] Failed to load model: {e}")
            raise

    def _load_cache_from_disk(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"predicate_embeddings_{self.embedding_model_name.replace('/', '_')}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"[PredicateMatcher] Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"[PredicateMatcher] Failed to load cache: {e}")
                self._embedding_cache = {}

    def _save_cache_to_disk(self):
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / f"predicate_embeddings_{self.embedding_model_name.replace('/', '_')}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self._embedding_cache, f)
            logger.debug(f"[PredicateMatcher] Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"[PredicateMatcher] Failed to save cache: {e}")

    @staticmethod
    def _expand_predicate_name(pred_name: str) -> str:
        """
        Expand predicate name to natural language.

        Examples:
            "birthDate" → "birth date"
            "date_of_birth" → "date of birth"
            "dbo:Person/name" → "person name"
            "foaf:knows" → "knows"

        Args:
            pred_name: Predicate local name

        Returns:
            Expanded natural language string
        """
        # Remove namespace prefixes
        if ":" in pred_name:
            pred_name = pred_name.split(":")[-1]
        if "/" in pred_name:
            pred_name = pred_name.split("/")[-1]

        # Convert camelCase to spaces
        # birthDate → birth Date
        expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', pred_name)

        # Convert snake_case and kebab-case to spaces
        expanded = re.sub(r'[_-]', ' ', expanded)

        # Lowercase and normalize whitespace
        expanded = re.sub(r'\s+', ' ', expanded.lower()).strip()

        return expanded

    def _get_embedding(self, predicate: str) -> np.ndarray:
        """
        Get embedding for a predicate, using cache if available.

        Args:
            predicate: Predicate local name

        Returns:
            Embedding vector
        """
        # Check cache
        if predicate in self._embedding_cache:
            return self._embedding_cache[predicate]

        # Initialize model if needed
        self._init_model()

        # Expand predicate name for better semantic understanding
        expanded = self._expand_predicate_name(predicate)

        # Compute embedding
        embedding = self.model.encode(expanded, convert_to_numpy=True, show_progress_bar=False)

        # Cache it
        self._embedding_cache[predicate] = embedding

        return embedding

    def _get_embedding_with_attr_names(
        self,
        predicate_local_name: str,
        predicate_uri: URIRef,
        attr_names: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Get embedding for a predicate, using attr_names if available.

        Priority:
        1. Use natural name from attr_names if available
        2. Fallback to automatic expansion of local name

        Args:
            predicate_local_name: Local name (e.g., "birthDate")
            predicate_uri: Full URI (e.g., URIRef("dbo:birthDate"))
            attr_names: Optional mapping of URI -> natural name

        Returns:
            Embedding vector
        """
        # Try to find natural name in attr_names
        natural_name = None
        if attr_names:
            uri_str = str(predicate_uri)
            natural_name = attr_names.get(uri_str)

        # Use natural name if found, otherwise use local name with expansion
        text_to_embed = natural_name if natural_name else predicate_local_name

        # Create cache key that includes source (attr_names vs expansion)
        cache_key = f"attr:{text_to_embed}" if natural_name else f"exp:{predicate_local_name}"

        # Check cache
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Initialize model if needed
        self._init_model()

        # Expand if not using attr_names
        if not natural_name:
            text_to_embed = self._expand_predicate_name(text_to_embed)

        # Compute embedding
        embedding = self.model.encode(text_to_embed, convert_to_numpy=True, show_progress_bar=False)

        # Cache it
        self._embedding_cache[cache_key] = embedding

        return embedding

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity in [0, 1]
        """
        # Normalize
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        # Clamp to [0, 1] (should already be, but just in case)
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity

    def match_predicates(
        self,
        src_predicates: Dict[str, Tuple[URIRef, List[Literal]]],
        tgt_predicates: Dict[str, Tuple[URIRef, List[Literal]]],
        src_attr_names: Optional[Dict[str, str]] = None,
        tgt_attr_names: Optional[Dict[str, str]] = None,
    ) -> List[PredicateMatch]:
        """
        Match predicates between source and target using semantic similarity.

        Args:
            src_predicates: Dict mapping local_name -> (uri, [literals])
            tgt_predicates: Dict mapping local_name -> (uri, [literals])
            src_attr_names: Optional dict mapping predicate URI -> natural name
            tgt_attr_names: Optional dict mapping predicate URI -> natural name

        Returns:
            List of PredicateMatch objects sorted by confidence (descending)
        """
        if not src_predicates or not tgt_predicates:
            return []

        logger.info(f"[PredicateMatcher] Matching {len(src_predicates)} source ↔ {len(tgt_predicates)} target predicates")

        # Count how many predicates have attr_names
        src_with_attr = sum(1 for name in src_predicates.keys()
                           if src_attr_names and str(src_predicates[name][0]) in src_attr_names)
        tgt_with_attr = sum(1 for name in tgt_predicates.keys()
                           if tgt_attr_names and str(tgt_predicates[name][0]) in tgt_attr_names)

        if src_attr_names and src_with_attr > 0:
            logger.info(f"[PredicateMatcher] Source: {src_with_attr}/{len(src_predicates)} predicates using attr_names, "
                       f"{len(src_predicates) - src_with_attr} using expansion")
        else:
            logger.info(f"[PredicateMatcher] Source: all {len(src_predicates)} predicates using automatic expansion")

        if tgt_attr_names and tgt_with_attr > 0:
            logger.info(f"[PredicateMatcher] Target: {tgt_with_attr}/{len(tgt_predicates)} predicates using attr_names, "
                       f"{len(tgt_predicates) - tgt_with_attr} using expansion")
        else:
            logger.info(f"[PredicateMatcher] Target: all {len(tgt_predicates)} predicates using automatic expansion")

        # Compute embeddings for all predicates
        src_names = list(src_predicates.keys())
        tgt_names = list(tgt_predicates.keys())

        # Use attr_names if available, otherwise use expansion
        src_embeddings = np.array([
            self._get_embedding_with_attr_names(name, src_predicates[name][0], src_attr_names)
            for name in src_names
        ])
        tgt_embeddings = np.array([
            self._get_embedding_with_attr_names(name, tgt_predicates[name][0], tgt_attr_names)
            for name in tgt_names
        ])

        # Compute similarity matrix
        # Shape: (num_src, num_tgt)
        similarity_matrix = self._compute_similarity_matrix(src_embeddings, tgt_embeddings)

        # Find ALL matches above threshold (many-to-many matching)
        matches = []

        # For each source predicate, find ALL target matches above threshold
        for i, src_name in enumerate(src_names):
            src_matches = []

            for j, tgt_name in enumerate(tgt_names):
                sim = similarity_matrix[i, j]

                if sim >= self.similarity_threshold:
                    match = PredicateMatch(
                        src_predicate=src_name,
                        tgt_predicate=tgt_name,
                        src_uri=src_predicates[src_name][0],
                        tgt_uri=tgt_predicates[tgt_name][0],
                        confidence=sim,
                        strategy="semantic_embedding",
                    )
                    src_matches.append(match)

            # Log all matches for this source predicate
            if src_matches:
                if len(src_matches) > 1:
                    logger.debug(f"  {src_name} → {len(src_matches)} matches: " +
                               ", ".join(f"{m.tgt_predicate}({m.confidence:.2f})" for m in src_matches))
                else:
                    logger.debug(f"  {src_name} ↔ {src_matches[0].tgt_predicate} (confidence: {src_matches[0].confidence:.3f})")

                matches.extend(src_matches)

        # Sort by confidence (descending)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        # Log summary
        if matches:
            avg_conf = sum(m.confidence for m in matches) / len(matches)
            high_conf = sum(1 for m in matches if m.confidence >= 0.85)

            # Count unique source and target predicates involved
            unique_src = len(set(m.src_predicate for m in matches))
            unique_tgt = len(set(m.tgt_predicate for m in matches))

            logger.info(f"[PredicateMatcher] ✓ Found {len(matches)} matches ({unique_src} src → {unique_tgt} tgt, "
                       f"avg conf: {avg_conf:.3f}, {high_conf} excellent ≥0.85)")
        else:
            logger.info(f"[PredicateMatcher] No matches found above threshold {self.similarity_threshold}")
            # Show best candidates that didn't make the cut (for debugging)
            if similarity_matrix.size > 0:
                max_sim = similarity_matrix.max()
                max_idx = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
                logger.info(f"[PredicateMatcher] Best candidate: {src_names[max_idx[0]]} ↔ {tgt_names[max_idx[1]]} "
                           f"(similarity: {max_sim:.3f}, threshold is {self.similarity_threshold})")

        # Save cache to disk (async would be better, but keep it simple)
        if len(self._embedding_cache) > 0:
            self._save_cache_to_disk()

        return matches

    def _compute_similarity_matrix(
        self,
        src_embeddings: np.ndarray,
        tgt_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            src_embeddings: Shape (num_src, embedding_dim)
            tgt_embeddings: Shape (num_tgt, embedding_dim)

        Returns:
            Similarity matrix of shape (num_src, num_tgt)
        """
        # Normalize embeddings
        src_norm = src_embeddings / (np.linalg.norm(src_embeddings, axis=1, keepdims=True) + 1e-8)
        tgt_norm = tgt_embeddings / (np.linalg.norm(tgt_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute dot product (cosine similarity)
        similarity_matrix = np.dot(src_norm, tgt_norm.T)

        # Clamp to [0, 1]
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        return similarity_matrix

    def compute_match_statistics(
        self,
        matches: List[PredicateMatch]
    ) -> Dict[str, Any]:
        """
        Compute statistics about matches.

        Args:
            matches: List of predicate matches

        Returns:
            Dictionary with statistics
        """
        if not matches:
            return {
                "num_matches": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        confidences = [m.confidence for m in matches]

        return {
            "num_matches": len(matches),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "std_confidence": np.std(confidences),
        }

    def visualize_matches(
        self,
        matches: List[PredicateMatch],
        top_k: int = 10
    ) -> str:
        """
        Create a human-readable visualization of matches.

        Args:
            matches: List of predicate matches
            top_k: Show only top K matches

        Returns:
            Formatted string
        """
        if not matches:
            return "No matches found."

        lines = [f"\n{'='*70}"]
        lines.append(f"Top {min(top_k, len(matches))} Predicate Matches")
        lines.append(f"{'='*70}\n")

        for i, match in enumerate(matches[:top_k], 1):
            src_expanded = self._expand_predicate_name(match.src_predicate)
            tgt_expanded = self._expand_predicate_name(match.tgt_predicate)

            lines.append(f"{i}. [{match.confidence:.3f}]")
            lines.append(f"   Source: {match.src_predicate:20s} → \"{src_expanded}\"")
            lines.append(f"   Target: {match.tgt_predicate:20s} → \"{tgt_expanded}\"")
            lines.append("")

        stats = self.compute_match_statistics(matches)
        lines.append(f"{'='*70}")
        lines.append(f"Statistics:")
        lines.append(f"  Total matches: {stats['num_matches']}")
        lines.append(f"  Avg confidence: {stats['avg_confidence']:.3f}")
        lines.append(f"  Min confidence: {stats['min_confidence']:.3f}")
        lines.append(f"  Max confidence: {stats['max_confidence']:.3f}")
        lines.append(f"{'='*70}\n")

        return "\n".join(lines)

    def clear_cache(self):
        """Clear embedding cache from memory and disk."""
        self._embedding_cache.clear()

        cache_file = self.cache_dir / f"predicate_embeddings_{self.embedding_model_name.replace('/', '_')}.pkl"
        if cache_file.exists():
            cache_file.unlink()
            logger.info("[PredicateMatcher] Cache cleared")


# Convenience function for quick matching
def match_predicates_semantic(
    src_predicates: Dict[str, Tuple[URIRef, List[Literal]]],
    tgt_predicates: Dict[str, Tuple[URIRef, List[Literal]]],
    threshold: float = 0.7,
    model: str = "all-MiniLM-L6-v2",
) -> List[PredicateMatch]:
    """
    Convenience function for semantic predicate matching.

    Args:
        src_predicates: Source predicates
        tgt_predicates: Target predicates
        threshold: Minimum similarity threshold
        model: Sentence transformer model name

    Returns:
        List of matches
    """
    matcher = PredicateMatcher({
        "embedding_model": model,
        "similarity_threshold": threshold,
    })

    return matcher.match_predicates(src_predicates, tgt_predicates)
