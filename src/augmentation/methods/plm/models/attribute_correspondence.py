"""Domain model for attribute correspondences between knowledge graphs."""

from dataclasses import dataclass
from typing import List
from rdflib import URIRef


@dataclass
class AttributeCorrespondence:
    """Represents a correspondence between source and target attributes.

    This is a cleaner, more OOP version of PredicateAlignment.
    """

    src_uri: URIRef
    tgt_uri: URIRef
    confidence: float
    source: str  # 'ground_truth' or 'semantic'

    # Similarity components (for semantic matches)
    name_similarity: float = 0.0
    value_similarity: float = 0.0

    # Sample values for debugging
    src_sample_values: List[str] = None
    tgt_sample_values: List[str] = None

    def __post_init__(self):
        """Initialize sample values if not provided."""
        if self.src_sample_values is None:
            self.src_sample_values = []
        if self.tgt_sample_values is None:
            self.tgt_sample_values = []

    @property
    def is_ground_truth(self) -> bool:
        """Check if this correspondence comes from ground-truth."""
        return self.source == 'ground_truth'

    @property
    def is_semantic(self) -> bool:
        """Check if this correspondence was discovered semantically."""
        return self.source == 'semantic'

    def __repr__(self) -> str:
        return (
            f"AttributeCorrespondence("
            f"{self.src_uri} ↔ {self.tgt_uri}, "
            f"confidence={self.confidence:.3f}, "
            f"source={self.source})"
        )
