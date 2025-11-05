"""PLM-based augmentation strategy leveraging latent interpolation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import torch
from rdflib import Literal, URIRef
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.augmentation.base import AugmentationMethod
from src.augmentation.methods.plm.interpolator import BartInterpolatorPLM, _clean_pred
from src.augmentation.methods.plm.predicate_matching import match_relations
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via BART-based latent interpolation of literal attributes."""

    registry_name = "plm_augmentation"

    def __init__(self, config):
        super().__init__(config)

    def augment(self, dataset: Dataset) -> Dataset:
        pass
