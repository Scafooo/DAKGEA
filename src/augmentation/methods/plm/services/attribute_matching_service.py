"""Service for computing attribute correspondences between knowledge graphs.

This service encapsulates the logic from predicate_alignment.py in a cleaner OOP interface.
"""

import logging
from typing import Dict, List, Optional

from src.core.dataset import Dataset
from src.logger import get_logger

from ..models import AttributeCorrespondence
from ..predicate_alignment import PredicateAlignmentCache, PredicateAlignment

logger = get_logger(__name__)


class AttributeMatchingService:
    """Service for computing and managing attribute correspondences.

    This service wraps the existing PredicateAlignmentCache but provides a
    cleaner interface using the new AttributeCorrespondence domain model.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the attribute matching service.

        Args:
            config: Configuration dictionary for matching behavior
        """
        self.config = config or {}
        predicate_matcher_config = self.config.get("predicate_matching", {})

        # Check if value-based matching is enabled
        self.use_value_similarity = predicate_matcher_config.get(
            "use_value_similarity", False
        )

        # Initialize cache if value-based matching is enabled
        self.cache: Optional[PredicateAlignmentCache] = None
        if self.use_value_similarity:
            self._init_alignment_cache(predicate_matcher_config)

    def _init_alignment_cache(self, config: Dict) -> None:
        """Initialize the predicate alignment cache."""
        name_weight = config.get("name_weight", 0.7)
        value_weight = config.get("value_weight", 0.3)
        sample_size = config.get("alignment_sample_size", 100)

        logger.info(
            f"[AttributeMatching] Initializing cache "
            f"(name_weight={name_weight}, value_weight={value_weight}, "
            f"sample_size={sample_size})"
        )

        self.cache = PredicateAlignmentCache(
            predicate_matcher_config=config,
            name_weight=name_weight,
            value_weight=value_weight,
            sample_size=sample_size,
        )

    def compute_correspondences(
        self, dataset: Dataset
    ) -> List[AttributeCorrespondence]:
        """Compute attribute correspondences for a dataset.

        This combines ground-truth matches (from match_attr files) with
        semantic matches computed from name and value similarity.

        Args:
            dataset: The dataset to compute correspondences for

        Returns:
            List of AttributeCorrespondence objects
        """
        if not self.use_value_similarity or self.cache is None:
            logger.info("[AttributeMatching] Value-based matching disabled")
            return self._get_ground_truth_correspondences(dataset)

        logger.info("[AttributeMatching] Computing correspondences...")

        # Compute alignments (includes both ground-truth and semantic)
        alignments = self.cache.compute_alignments(dataset)

        # Convert to AttributeCorrespondence objects
        correspondences = [
            self._convert_alignment(alignment) for alignment in alignments
        ]

        logger.info(
            f"[AttributeMatching] Computed {len(correspondences)} correspondences"
        )

        return correspondences

    def _get_ground_truth_correspondences(
        self, dataset: Dataset
    ) -> List[AttributeCorrespondence]:
        """Get only ground-truth correspondences from dataset."""
        correspondences = []
        ground_truth = dataset.attribute_matches or {}

        for src_uri_str, tgt_uri_list in ground_truth.items():
            for tgt_uri_str in tgt_uri_list:
                from rdflib import URIRef

                correspondence = AttributeCorrespondence(
                    src_uri=URIRef(src_uri_str),
                    tgt_uri=URIRef(tgt_uri_str),
                    confidence=1.0,
                    source="ground_truth",
                    name_similarity=1.0,
                    value_similarity=1.0,
                )
                correspondences.append(correspondence)

        logger.info(
            f"[AttributeMatching] Loaded {len(correspondences)} ground-truth correspondences"
        )
        return correspondences

    def _convert_alignment(
        self, alignment: PredicateAlignment
    ) -> AttributeCorrespondence:
        """Convert PredicateAlignment to AttributeCorrespondence."""
        # Determine source based on name_similarity
        # Ground-truth matches have name_similarity = 1.0
        is_ground_truth = alignment.name_similarity >= 0.999
        source = "ground_truth" if is_ground_truth else "semantic"

        return AttributeCorrespondence(
            src_uri=alignment.src_uri,
            tgt_uri=alignment.tgt_uri,
            confidence=alignment.combined_score,
            source=source,
            name_similarity=alignment.name_similarity,
            value_similarity=alignment.value_similarity,
            src_sample_values=alignment.src_sample_values,
            tgt_sample_values=alignment.tgt_sample_values,
        )

    def get_correspondences_for_source(
        self, src_uri
    ) -> List[AttributeCorrespondence]:
        """Get all correspondences for a source attribute.

        Args:
            src_uri: Source attribute URI

        Returns:
            List of correspondences for the source attribute
        """
        if self.cache is None:
            return []

        alignments = self.cache.get_alignments_for_source(src_uri)
        return [self._convert_alignment(a) for a in alignments]

    def get_best_correspondence(self, src_uri) -> Optional[AttributeCorrespondence]:
        """Get the best correspondence for a source attribute.

        Args:
            src_uri: Source attribute URI

        Returns:
            Best correspondence or None if no match found
        """
        if self.cache is None:
            return None

        best_alignment = self.cache.get_best_match(src_uri)
        if best_alignment is None:
            return None

        return self._convert_alignment(best_alignment)
