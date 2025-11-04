"""BERT-INT Interaction Model - Phase 2 of BERT-INT pipeline."""

from .features import (
    DualAggregation,
    NeighborViewFeatureExtractor,
    AttributeViewFeatureExtractor,
    DescriptionViewFeatureExtractor,
)
from .model import InteractionMLP
from .dataset import InteractionDataset, CandidateGenerator, AttributeValueCleaner
from .trainer import InteractionTrainer
from .evaluator import InteractionEvaluator

__all__ = [
    "DualAggregation",
    "NeighborViewFeatureExtractor",
    "AttributeViewFeatureExtractor",
    "DescriptionViewFeatureExtractor",
    "InteractionMLP",
    "InteractionDataset",
    "CandidateGenerator",
    "AttributeValueCleaner",
    "InteractionTrainer",
    "InteractionEvaluator",
]
