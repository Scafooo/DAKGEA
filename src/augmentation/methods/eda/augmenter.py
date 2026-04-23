"""EDA (Easy Data Augmentation) augmentation method for KG entity resolution."""

from __future__ import annotations

import math
import random
from typing import Optional

from rdflib import Literal, URIRef, XSD

from src.augmentation.base import AugmentationMethod
from src.augmentation.methods.eda.eda_core import eda
from src.augmentation.registry import AUGMENTATION_REGISTRY


@AUGMENTATION_REGISTRY.register("eda")
class EDAAugmenter(AugmentationMethod):
    """
    Augments KG training pairs by applying EDA text transformations
    (synonym replacement, random insertion, random swap, random deletion)
    to string-valued entity attributes, producing synthetic matched pairs.
    """

    registry_name = "eda"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        aug_cfg = self.config.get("augmentation", {})
        self.ratio: float = float(aug_cfg.get("ratio", 0.3))
        self.max_pairs: Optional[int] = aug_cfg.get("max_pairs", None)
        self.alpha_sr: float = float(aug_cfg.get("alpha_sr", 0.1))
        self.alpha_ri: float = float(aug_cfg.get("alpha_ri", 0.1))
        self.alpha_rs: float = float(aug_cfg.get("alpha_rs", 0.1))
        self.alpha_rd: float = float(aug_cfg.get("alpha_rd", 0.1))

    def augment(self, dataset) -> object:
        self.section("EDA Augmentation")

        aligned = list(dataset.aligned_entities)
        budget = (
            self.max_pairs
            if self.max_pairs is not None
            else math.floor(len(aligned) * self.ratio)
        )
        self.logger.info(
            "Budget: %d synthetic pairs (ratio=%.2f, seed pairs=%d)",
            budget, self.ratio, len(aligned),
        )

        seeds = aligned.copy()
        random.shuffle(seeds)

        generated = 0
        for src_uri, tgt_uri in seeds:
            if generated >= budget:
                break

            src_aug = URIRef(f"{src_uri}_eda_{generated}")
            tgt_aug = URIRef(f"{tgt_uri}_eda_{generated}")

            self._augment_entity(
                dataset.knowledge_graph_source, src_uri, src_aug
            )
            self._augment_entity(
                dataset.knowledge_graph_target, tgt_uri, tgt_aug
            )

            aligned.append((str(src_aug), str(tgt_aug)))
            generated += 1

        dataset.aligned_entities = tuple(aligned)
        self.logger.info("Added %d synthetic pairs.", generated)
        return dataset

    # ------------------------------------------------------------------

    def _augment_entity(self, graph, original_uri: str, aug_uri: URIRef) -> None:
        """Copy all triples of *original_uri* to *aug_uri*, applying EDA to string literals."""
        for _, pred, obj in graph.triples((URIRef(original_uri), None, None)):
            if self._is_string_literal(obj):
                variants = eda(
                    str(obj),
                    alpha_sr=self.alpha_sr,
                    alpha_ri=self.alpha_ri,
                    alpha_rs=self.alpha_rs,
                    p_rd=self.alpha_rd,
                    num_aug=1,
                )
                # variants[-1] is the original; variants[0] is the augmented one
                aug_val = variants[0] if len(variants) > 1 else str(obj)
                graph.add((aug_uri, pred, Literal(aug_val)))
            else:
                graph.add((aug_uri, pred, obj))

    @staticmethod
    def _is_string_literal(obj) -> bool:
        """Return True for plain or xsd:string literals worth augmenting."""
        if not isinstance(obj, Literal):
            return False
        if obj.datatype is None:
            return True
        return obj.datatype in (XSD.string, XSD.normalizedString)
