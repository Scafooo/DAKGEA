"""Pre-computed predicate alignment for PLM augmentation.

This module handles one-time computation of predicate alignments between
source and target KGs, combining semantic name similarity with value similarity.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger
from .predicate_matcher import PredicateMatcher, PredicateMatch

logger = get_logger(__name__)


@dataclass
class PredicateAlignment:
    """Represents a pre-computed alignment between source and target predicates."""
    src_uri: URIRef
    tgt_uri: URIRef
    name_similarity: float  # Semantic similarity of predicate names
    value_similarity: float  # Similarity of value distributions
    combined_score: float  # Final hybrid score
    src_sample_values: List[str]  # Sample values for debugging
    tgt_sample_values: List[str]  # Sample values for debugging


class PredicateAlignmentCache:
    """Pre-computes and caches predicate alignments for efficient lookup."""

    def __init__(
        self,
        predicate_matcher_config: Optional[Dict] = None,
        name_weight: float = 0.7,  # Weight for name similarity
        value_weight: float = 0.3,  # weight for value similarity
        min_samples: int = 3,  # Minimum samples to compute value similarity
        sample_size: int = 100,  # Number of entities to sample for value collection
    ):
        """Initialize the alignment cache.

        Args:
            predicate_matcher_config: Config for the semantic matcher
            name_weight: Weight for name similarity (alpha)
            value_weight: Weight for value similarity (1-alpha)
            min_samples: Minimum samples needed to compute value similarity
            sample_size: Number of entities to sample per predicate
        """
        self.predicate_matcher = PredicateMatcher(predicate_matcher_config or {})
        self.name_weight = name_weight
        self.value_weight = value_weight
        self.min_samples = min_samples
        self.sample_size = sample_size

        # Cache: (src_uri, tgt_uri) -> PredicateAlignment
        self._alignment_cache: Dict[Tuple[str, str], PredicateAlignment] = {}

        # Quick lookup by local name
        self._by_local_name: Dict[Tuple[str, str], PredicateAlignment] = {}

    def compute_alignments(
        self,
        dataset: Dataset,
        sample_entities: Optional[int] = None,
    ) -> List[PredicateAlignment]:
        """Pre-compute all predicate alignments for the dataset.

        Args:
            dataset: The dataset to compute alignments for
            sample_entities: Number of entities to sample (None = use config default or all)

        Returns:
            List of PredicateAlignment objects
        """
        logger.info("[PredicateAlignment] Pre-computing predicate alignments...")

        # Auto-adjust sample size for small datasets
        num_aligned = len(dataset.aligned_entities)
        if sample_entities is None:
            # Use all entities if dataset is small, otherwise use configured sample_size
            sample_entities = min(num_aligned, self.sample_size)
            logger.info(f"[PredicateAlignment] Auto-adjusted sample size to {sample_entities} (dataset has {num_aligned} aligned pairs)")

        # Also adjust min_samples for very small datasets
        original_min_samples = self.min_samples
        if sample_entities < 10:
            self.min_samples = max(1, sample_entities // 3)
            logger.info(f"[PredicateAlignment] Adjusted min_samples from {original_min_samples} to {self.min_samples} for small dataset")

        # Step 1: Collect predicate value samples from both KGs
        logger.info("[PredicateAlignment] Collecting predicate samples from source KG...")
        src_samples = self._collect_predicate_samples(
            dataset.knowledge_graph_source,
            sample_entities or self.sample_size
        )
        logger.info(f"[PredicateAlignment] Collected samples for {len(src_samples)} source predicates")

        logger.info("[PredicateAlignment] Collecting predicate samples from target KG...")
        tgt_samples = self._collect_predicate_samples(
            dataset.knowledge_graph_target,
            sample_entities or self.sample_size
        )
        logger.info(f"[PredicateAlignment] Collected samples for {len(tgt_samples)} target predicates")

        # Step 2: Compute semantic name similarity
        logger.info("[PredicateAlignment] Computing semantic name similarities...")

        # Prepare data for PredicateMatcher
        src_predicates = {
            local_name: (uri, values)
            for local_name, (uri, values) in src_samples.items()
        }
        tgt_predicates = {
            local_name: (uri, values)
            for local_name, (uri, values) in tgt_samples.items()
        }

        # Get attr_names
        src_attr_names = dataset.knowledge_graph_source.attr_to_name
        tgt_attr_names = dataset.knowledge_graph_target.attr_to_name

        # Compute name-based matches with very low threshold for pre-computation
        # We'll filter by combined_score later
        original_threshold = self.predicate_matcher.similarity_threshold
        self.predicate_matcher.similarity_threshold = 0.3  # Very permissive during pre-computation
        logger.info(f"[PredicateAlignment] Using permissive threshold {self.predicate_matcher.similarity_threshold} for pre-computation")

        name_matches = self.predicate_matcher.match_predicates(
            src_predicates,
            tgt_predicates,
            src_attr_names,
            tgt_attr_names,
        )

        # Restore original threshold
        self.predicate_matcher.similarity_threshold = original_threshold

        logger.info(f"[PredicateAlignment] Found {len(name_matches)} name-based matches (before value similarity filtering)")

        # Step 3: Enhance with value similarity
        logger.info("[PredicateAlignment] Computing value similarities...")
        alignments = []

        for match in name_matches:
            src_uri = match.src_uri
            tgt_uri = match.tgt_uri
            src_values = src_samples[match.src_predicate][1]
            tgt_values = tgt_samples[match.tgt_predicate][1]

            # Compute value similarity
            value_sim = self._compute_value_similarity(src_values, tgt_values)

            # Combine scores
            combined = (
                self.name_weight * match.confidence +
                self.value_weight * value_sim
            )

            alignment = PredicateAlignment(
                src_uri=src_uri,
                tgt_uri=tgt_uri,
                name_similarity=match.confidence,
                value_similarity=value_sim,
                combined_score=combined,
                src_sample_values=[str(v)[:30] for v in src_values[:3]],
                tgt_sample_values=[str(v)[:30] for v in tgt_values[:3]],
            )
            alignments.append(alignment)

            # Cache by URI
            self._alignment_cache[(str(src_uri), str(tgt_uri))] = alignment
            # Cache by local name
            self._by_local_name[(match.src_predicate, match.tgt_predicate)] = alignment

        # Sort by combined score (best matches first)
        alignments.sort(key=lambda a: a.combined_score, reverse=True)

        # Keep all alignments - filtering will be done during usage based on context
        # This allows the system to use even weaker matches when needed

        logger.info(f"[PredicateAlignment] ✓ Pre-computed {len(alignments)} alignments")
        if alignments:
            logger.info(f"[PredicateAlignment] Average combined score: {np.mean([a.combined_score for a in alignments]):.3f}")

        # Show top alignments
        if alignments:
            logger.info("[PredicateAlignment] Top 5 alignments:")
            for i, align in enumerate(alignments[:5], 1):
                logger.info(f"  {i}. {align.src_uri} ↔ {align.tgt_uri}")
                logger.info(f"     name_sim={align.name_similarity:.3f}, value_sim={align.value_similarity:.3f}, "
                           f"combined={align.combined_score:.3f}")

        return alignments

    def get_alignment(
        self,
        src_predicate: URIRef,
        tgt_predicate: URIRef,
    ) -> Optional[PredicateAlignment]:
        """Get pre-computed alignment for a predicate pair.

        Args:
            src_predicate: Source predicate URI
            tgt_predicate: Target predicate URI

        Returns:
            PredicateAlignment if found, None otherwise
        """
        key = (str(src_predicate), str(tgt_predicate))
        return self._alignment_cache.get(key)

    def get_alignments_for_source(self, src_predicate: URIRef) -> List[PredicateAlignment]:
        """Get all alignments for a source predicate.

        Args:
            src_predicate: Source predicate URI

        Returns:
            List of alignments, sorted by combined score
        """
        src_str = str(src_predicate)
        alignments = [
            align for (src, tgt), align in self._alignment_cache.items()
            if src == src_str
        ]
        alignments.sort(key=lambda a: a.combined_score, reverse=True)
        return alignments

    def get_best_match(self, src_predicate: URIRef) -> Optional[PredicateAlignment]:
        """Get the best alignment for a source predicate.

        Args:
            src_predicate: Source predicate URI

        Returns:
            Best PredicateAlignment or None
        """
        alignments = self.get_alignments_for_source(src_predicate)
        return alignments[0] if alignments else None

    def _collect_predicate_samples(
        self,
        graph,
        max_entities: int,
    ) -> Dict[str, Tuple[URIRef, List[Literal]]]:
        """Collect sample values for each predicate in the graph.

        Args:
            graph: KnowledgeGraph instance
            max_entities: Maximum entities to sample

        Returns:
            Dict mapping predicate_local_name -> (uri, [literal_values])
        """
        from .bart_interpolator import _clean_pred

        predicate_samples = {}
        entity_count = 0
        seen_entities = set()

        for subject, predicate, obj in graph.triples((None, None, None)):
            if isinstance(obj, Literal):
                # Track entities seen (for statistics)
                if subject not in seen_entities:
                    seen_entities.add(subject)
                    entity_count += 1

                # Extract local name
                local_name = _clean_pred(str(predicate))

                # Initialize if needed
                if local_name not in predicate_samples:
                    predicate_samples[local_name] = (predicate, [])

                # Add sample value (limit per predicate to avoid memory issues)
                if len(predicate_samples[local_name][1]) < self.sample_size:
                    predicate_samples[local_name][1].append(obj)

                # Early exit if all predicates have enough samples
                if entity_count > max_entities:
                    # Check if all predicates have enough samples
                    all_full = all(len(vals) >= self.sample_size for _, vals in predicate_samples.values())
                    if all_full:
                        break

        return predicate_samples

    def _compute_value_similarity(
        self,
        src_values: List[Literal],
        tgt_values: List[Literal],
    ) -> float:
        """Compute similarity between two sets of literal values.

        Uses multiple heuristics:
        1. Type compatibility (dates, numbers, text)
        2. String overlap (Jaccard similarity for text)
        3. Numeric correlation (for numbers)
        4. Exact match ratio

        Args:
            src_values: Source literal values
            tgt_values: Target literal values

        Returns:
            Similarity score in [0, 1]
        """
        if not src_values or not tgt_values:
            logger.debug(f"Value similarity: empty values (src={len(src_values) if src_values else 0}, tgt={len(tgt_values) if tgt_values else 0})")
            return 0.0

        if len(src_values) < self.min_samples or len(tgt_values) < self.min_samples:
            logger.debug(f"Value similarity: insufficient samples (src={len(src_values)}, tgt={len(tgt_values)}, min={self.min_samples})")
            return 0.0

        # Convert to strings
        src_strs = [str(v).lower().strip() for v in src_values[:self.sample_size]]
        tgt_strs = [str(v).lower().strip() for v in tgt_values[:self.sample_size]]

        # Heuristic 1: Type compatibility
        src_type = self._infer_value_type(src_strs)
        tgt_type = self._infer_value_type(tgt_strs)

        if src_type != tgt_type and src_type != "mixed" and tgt_type != "mixed":
            # Incompatible types
            return 0.1

        # Heuristic 2: Exact match ratio
        src_set = set(src_strs)
        tgt_set = set(tgt_strs)
        exact_matches = len(src_set & tgt_set)
        exact_ratio = exact_matches / min(len(src_set), len(tgt_set)) if src_set and tgt_set else 0.0

        # Heuristic 3: Jaccard similarity (for text)
        if src_type == "text" and tgt_type == "text":
            # Token-level Jaccard
            src_tokens = set(' '.join(src_strs).split())
            tgt_tokens = set(' '.join(tgt_strs).split())
            jaccard = len(src_tokens & tgt_tokens) / len(src_tokens | tgt_tokens) if src_tokens | tgt_tokens else 0.0
            return 0.4 * exact_ratio + 0.6 * jaccard

        # Heuristic 4: Numeric similarity (for numbers/dates)
        if src_type in ("number", "date") and tgt_type in ("number", "date"):
            # Try to extract numbers
            src_nums = self._extract_numbers(src_strs)
            tgt_nums = self._extract_numbers(tgt_strs)

            if src_nums and tgt_nums:
                # Compute correlation or overlap
                src_mean = np.mean(src_nums)
                tgt_mean = np.mean(tgt_nums)
                src_std = np.std(src_nums) + 1e-8
                tgt_std = np.std(tgt_nums) + 1e-8

                # Normalized difference
                mean_diff = abs(src_mean - tgt_mean) / max(abs(src_mean), abs(tgt_mean), 1.0)
                std_ratio = min(src_std, tgt_std) / max(src_std, tgt_std)

                numeric_sim = (1 - mean_diff) * std_ratio
                return 0.5 * exact_ratio + 0.5 * numeric_sim

        # Default: use exact match ratio
        return exact_ratio

    @staticmethod
    def _infer_value_type(values: List[str]) -> str:
        """Infer the type of values (date, number, text, ID, mixed)."""
        if not values:
            return "unknown"

        types = []
        for val in values[:20]:  # Sample first 20
            if not val:
                continue

            # Check for date patterns
            if any(sep in val for sep in ['-', '/']):
                parts = val.replace('/', '-').split('-')
                if len(parts) >= 2 and any(p.isdigit() and len(p) >= 4 for p in parts):
                    types.append("date")
                    continue

            # Check for numbers
            try:
                float(val.replace(',', ''))
                types.append("number")
                continue
            except (ValueError, AttributeError):
                pass

            # Check for IDs (alphanumeric, short, no spaces)
            if len(val) < 20 and not ' ' in val and any(c.isdigit() for c in val):
                types.append("id")
                continue

            # Default: text
            types.append("text")

        if not types:
            return "unknown"

        # Majority vote
        from collections import Counter
        most_common = Counter(types).most_common(1)[0]

        # If no clear majority, return "mixed"
        if most_common[1] / len(types) < 0.6:
            return "mixed"

        return most_common[0]

    @staticmethod
    def _extract_numbers(values: List[str]) -> List[float]:
        """Extract numeric values from strings."""
        numbers = []
        for val in values:
            try:
                # Remove common separators
                cleaned = val.replace(',', '').replace(' ', '')
                # Try to extract first number
                import re
                match = re.search(r'-?\d+\.?\d*', cleaned)
                if match:
                    numbers.append(float(match.group()))
            except (ValueError, AttributeError):
                continue
        return numbers
