"""Random-entities reduction strategy."""

from __future__ import annotations

import random
from typing import Iterable, Set, Tuple

from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger
from src.reduction.registry import REDUCTION_REGISTRY

logger = get_logger(__name__)


@REDUCTION_REGISTRY.register("random_entities")
class RandomEntitiesReducer:
    """Sample aligned entity pairs uniformly at random and prune orphan triples."""

    def __init__(self, config):
        self.config = config
        reduction_cfg = self.config.get("reduction", {})
        self.target_entities = max(1, int(reduction_cfg.get("target_entities", 1)))
        self.seed = reduction_cfg.get("random_seed")
        self.filter_alignment = reduction_cfg.get("filter_alignment", True)
        if self.seed is None:
            experiment_cfg = self.config.get("experiment", {})
            self.seed = experiment_cfg.get("seed")

    def reduce(self, dataset: Dataset) -> Dataset:
        logger.info("[STEP] RandomEntities reduction started")
        aligned_set = self._normalise_alignment(dataset.aligned_entities)
        total_pairs = len(aligned_set)

        if total_pairs == 0:
            logger.warning("Dataset has no aligned entities; skipping reduction.")
            dataset.aligned_entities = aligned_set
            return dataset

        target_pairs = min(self.target_entities, total_pairs)
        remove_count = total_pairs - target_pairs

        logger.info(
            "Reducing dataset to %d aligned entity pairs (from %d) using random sampling.",
            target_pairs,
            total_pairs,
        )
        logger.debug("Random reduction seed: %s", self.seed)

        source_size_before = len(dataset.knowledge_graph_source)
        target_size_before = len(dataset.knowledge_graph_target)

        if remove_count > 0:
            to_remove = self._sample_pairs_to_remove(aligned_set, remove_count, self.seed)
            logger.debug("Selected %d alignment pairs for removal.", len(to_remove))
            self._prune_graphs(dataset, to_remove)
            remaining = aligned_set - to_remove
        else:
            logger.info(
                "Target pair count (%d) >= available pairs (%d); nothing to remove.",
                target_pairs,
                total_pairs,
            )
            remaining = aligned_set

        dataset.aligned_entities = remaining
        if self.filter_alignment:
            dataset.aligned_entities = self._filter_alignment(dataset)
        else:
            logger.debug("Alignment filtering disabled; retaining all sampled pairs.")

        source_size_after = len(dataset.knowledge_graph_source)
        target_size_after = len(dataset.knowledge_graph_target)

        logger.info(
            "Reduction complete. Source triples: %d → %d; target triples: %d → %d; aligned pairs: %d → %d.",
            source_size_before,
            source_size_after,
            target_size_before,
            target_size_after,
            total_pairs,
            len(dataset.aligned_entities),
        )
        logger.info("[SUCCESS] RandomEntities reduction finished")

        return dataset

    @staticmethod
    def _normalise_alignment(
        aligned_entities: Iterable[Tuple[object, object]]
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Convert aligned entity pairs to a set of URIRefs."""
        normalised: Set[Tuple[URIRef, URIRef]] = set()
        for left, right in aligned_entities:
            normalised.add(
                (
                    RandomEntitiesReducer._ensure_uri(left),
                    RandomEntitiesReducer._ensure_uri(right),
                )
            )
        return normalised

    @staticmethod
    def _ensure_uri(value) -> URIRef:
        if isinstance(value, URIRef):
            return value
        return URIRef(str(value))

    @staticmethod
    def _sample_pairs_to_remove(
        aligned_entities: Set[Tuple[URIRef, URIRef]],
        remove_count: int,
        seed: int | None = None,
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Randomly select a subset of alignment pairs to discard."""
        if remove_count <= 0:
            return set()

        ordered = sorted(
            aligned_entities,
            key=lambda pair: (str(pair[0]), str(pair[1])),
        )
        rng = random.Random(seed)
        return set(rng.sample(ordered, remove_count))

    @staticmethod
    def _prune_graphs(dataset: Dataset, pairs_to_remove: Set[Tuple[URIRef, URIRef]]) -> None:
        """Remove triples referencing entities slated for removal."""
        src_graph = dataset.knowledge_graph_source
        tgt_graph = dataset.knowledge_graph_target

        # Sort pairs to ensure deterministic iteration order
        for left, right in sorted(pairs_to_remove, key=lambda p: (str(p[0]), str(p[1]))):
            src_graph.remove((left, None, None))
            src_graph.remove((None, None, left))
            tgt_graph.remove((right, None, None))
            tgt_graph.remove((None, None, right))

    @staticmethod
    def _filter_alignment(dataset: Dataset) -> Set[Tuple[URIRef, URIRef]]:
        """Keep only aligned pairs whose entities remain with both relation and attribute triples."""
        source_relation_entities: Set[URIRef] = set()
        source_attribute_entities: Set[URIRef] = set()
        for subj, _, obj in dataset.knowledge_graph_source.triples((None, None, None)):
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                source_relation_entities.add(subj)
                source_relation_entities.add(obj)
            if isinstance(subj, URIRef) and isinstance(obj, Literal):
                source_attribute_entities.add(subj)

        target_relation_entities: Set[URIRef] = set()
        target_attribute_entities: Set[URIRef] = set()
        for subj, _, obj in dataset.knowledge_graph_target.triples((None, None, None)):
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                target_relation_entities.add(subj)
                target_relation_entities.add(obj)
            if isinstance(subj, URIRef) and isinstance(obj, Literal):
                target_attribute_entities.add(subj)

        filtered = {
            (left, right)
            for left, right in dataset.aligned_entities
            if left in source_relation_entities
            and left in source_attribute_entities
            and right in target_relation_entities
            and right in target_attribute_entities
        }

        return filtered


__all__ = ["RandomEntitiesReducer"]
