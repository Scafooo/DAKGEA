"""PLM-based augmentation strategy leveraging latent interpolation scaffolding."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from rdflib import Literal, URIRef

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset

from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via BART-based latent interpolation of literal attributes."""

    registry_name = "plm_augmentation"
    _DEFAULT_DERIVED_PREDICATE = URIRef("http://dakgea.org/augmentation/derivedFrom")

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        cfg = self.config or {}
        augmentation_cfg = cfg.get("augmentation", cfg)
        experiment_cfg = cfg.get("experiment", cfg)

        self.max_depth = int(augmentation_cfg.get("max_depth", 1))
        max_pairs = augmentation_cfg.get("max_pairs")
        self.max_pairs_config = max(1, int(max_pairs)) if max_pairs is not None else None

        ratio = augmentation_cfg.get("ratio")
        self.augmentation_ratio: Optional[float] = None
        if ratio is not None:
            try:
                ratio_value = float(ratio)
                if ratio_value > 0:
                    self.augmentation_ratio = ratio_value
                else:
                    self.logger.warning("augmentation.ratio must be > 0; ignoring '%s'.", ratio)
            except (TypeError, ValueError):
                self.logger.warning("Invalid augmentation.ratio '%s'; ignoring.", ratio)

        seed = experiment_cfg.get("seed", cfg.get("seed", 0))
        self.seed = int(seed) if seed is not None else 0

        derived_predicate = augmentation_cfg.get("derived_predicate", cfg.get("derived_predicate"))
        self.derived_predicate = URIRef(derived_predicate) if derived_predicate else self._DEFAULT_DERIVED_PREDICATE
        self._membership_cache: Dict[Tuple[str, URIRef], bool] = {}
        self._id_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def augment(self, dataset: Dataset) -> Dataset:
        """Augment a dataset by spawning aligned synthetic entities."""
        initial_pairs = len(dataset.aligned_entities)
        dataset_augmented = dataset.clone()
        if not dataset.aligned_entities:
            self.logger.warning("Skipping PLM augmentation: no aligned entities available.")
            return dataset_augmented

        set_graph = SetKnowledgeGraph.from_dataset(dataset)
        set_nodes = sorted(set_graph.iter_set_nodes(), key=lambda uri: str(uri))
        if not set_nodes:
            self.logger.warning("SetKnowledgeGraph is empty; nothing to augment.")
            return dataset_augmented

        pair_budget = self._compute_pair_budget(initial_pairs)
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        self.logger.info(
            "[PLM] Augmentation budget=%s (initial_pairs=%d, ratio=%s, max_pairs=%s, seed=%d)",
            budget_display,
            initial_pairs,
            f"{self.augmentation_ratio:.2f}" if self.augmentation_ratio is not None else "-",
            self.max_pairs_config if self.max_pairs_config is not None else "-",
            self.seed,
        )

        rng = random.Random(self.seed)
        rng.shuffle(set_nodes)

        self.section("PLM Augmentation")
        visited: set[URIRef] = set()
        expansion_chain: List[str] = []
        queue: Deque[Tuple[URIRef, int]] = deque()
        remaining_seeds: Deque[URIRef] = deque(set_nodes)
        expanded_pairs = 0
        self._membership_cache.clear()
        self._id_counter = 0

        while (queue or remaining_seeds) and (pair_budget is None or expanded_pairs < pair_budget):
            if not queue:
                while remaining_seeds and remaining_seeds[0] in visited:
                    remaining_seeds.popleft()
                if not remaining_seeds:
                    break
                queue.append((remaining_seeds.popleft(), 0))

            node, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            expansion_chain.append(str(node))

            self._log_node_connections(set_graph, node)

            src_aug, tgt_aug, src_original, tgt_original = self._spawn_augmented_pair(
                dataset_augmented, set_graph, node
            )
            expanded_pairs += 1

            self._bootstrap_literals(dataset_augmented, src_original, tgt_original, src_aug, tgt_aug)
            self._expansion(dataset_augmented, set_graph, node, src_aug, tgt_aug)

            if depth < self.max_depth:
                direct_enqueued: set[URIRef] = set()
                for direction, predicate, neighbor in self._iter_neighbors(set_graph, node):
                    if isinstance(neighbor, Literal):
                        self._attach_literal(dataset_augmented, predicate, neighbor, direction, src_aug, tgt_aug)
                        continue

                    if not isinstance(neighbor, URIRef):
                        continue

                    if set_graph.is_set_node(neighbor):
                        next_depth = depth + 1
                        if neighbor not in visited and neighbor not in direct_enqueued and next_depth <= self.max_depth:
                            queue.append((neighbor, next_depth))
                            direct_enqueued.add(neighbor)
                            self.logger.info("[PLM][Queue] direct set neighbor queued")
                            self.logger.info("    • current → %s", node)
                            self.logger.info("    • neighbor → %s", neighbor)
                            self.logger.info("    • depth → %d", next_depth)
                        continue

                    created_neighbors = self._augment_non_set_neighbor(
                        dataset_augmented, neighbor, predicate, direction, src_aug, tgt_aug
                    )
                    if created_neighbors:
                        expansion_chain.extend(str(uri) for uri in created_neighbors)
                    else:
                        expansion_chain.append(str(neighbor))

                    if depth + 1 > self.max_depth:
                        continue

                    bridged_enqueued: set[URIRef] = set()
                    for inner_direction, inner_predicate, inner_neighbor in self._iter_neighbors(set_graph, neighbor):
                        if not isinstance(inner_neighbor, URIRef):
                            continue
                        if not set_graph.is_set_node(inner_neighbor):
                            continue
                        if inner_neighbor is node:
                            continue
                        if inner_neighbor in visited:
                            continue
                        if inner_neighbor in bridged_enqueued:
                            continue
                        next_depth = depth + 2
                        if next_depth > self.max_depth:
                            continue
                        queue.append((inner_neighbor, next_depth))
                        bridged_enqueued.add(inner_neighbor)
                        self.logger.info("[PLM][Queue] bridged via non-set")
                        self.logger.info("    • current → %s", node)
                        self.logger.info("    • bridge → %s", neighbor)
                        self.logger.info("    • next → %s", inner_neighbor)
                        self.logger.info("    • depth → %d", next_depth)


        self.logger.info(
            "[PLM] Augmented %d/%s fused nodes (max_depth=%d).",
            expanded_pairs,
            budget_display,
            self.max_depth,
        )
        if expansion_chain:
            chain_repr = " -> ".join(expansion_chain)
            self.logger.info("[PLM] Expansion chain: %s", chain_repr)
        else:
            self.logger.info("[PLM] Expansion chain: (none)")
        return dataset_augmented

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_augmented_pair(
        self, dataset: Dataset, set_graph: SetKnowledgeGraph, set_node: URIRef
    ) -> Tuple[URIRef, URIRef, Optional[URIRef], Optional[URIRef]]:
        components = set_graph.get_components(set_node)
        src_component = components[0] if components else None
        tgt_component = components[1] if len(components) > 1 else None

        src_aug = self._mint_augmented_uri(src_component or set_node)
        tgt_aug = self._mint_augmented_uri(tgt_component or set_node)

        src_reference = src_component or set_node
        tgt_reference = tgt_component or set_node

        dataset.knowledge_graph_source.add((src_aug, self.derived_predicate, src_reference))
        dataset.knowledge_graph_target.add((tgt_aug, self.derived_predicate, tgt_reference))

        alignments = list(dataset.aligned_entities)
        alignments.append((str(src_aug), str(tgt_aug)))
        dataset.aligned_entities = tuple(alignments)

        return src_aug, tgt_aug, src_component, tgt_component

    def _bootstrap_literals(
        self,
        dataset: Dataset,
        src_component: Optional[URIRef],
        tgt_component: Optional[URIRef],
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        self._copy_literals(dataset.knowledge_graph_source, src_component, src_aug)
        self._copy_literals(dataset.knowledge_graph_target, tgt_component, tgt_aug)

    @staticmethod
    def _copy_literals(graph, original: Optional[URIRef], augmented: URIRef) -> None:
        if not original:
            return
        for _, predicate, obj in graph.triples((original, None, None)):
            if isinstance(obj, Literal):
                graph.add((augmented, predicate, obj))

    def _iter_neighbors(
        self, graph: SetKnowledgeGraph, node: URIRef
    ) -> Iterator[Tuple[str, URIRef, URIRef | Literal]]:
        for _, predicate, neighbor in graph.triples((node, None, None)):
            yield "out", predicate, neighbor
        for neighbor, predicate, _ in graph.triples((None, None, node)):
            yield "in", predicate, neighbor

    def _attach_literal(
        self,
        dataset: Dataset,
        predicate: URIRef,
        literal: Literal,
        direction: str,
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        if direction != "out":
            return
        dataset.knowledge_graph_source.add((src_aug, predicate, literal))
        dataset.knowledge_graph_target.add((tgt_aug, predicate, literal))

    def _augment_non_set_neighbor(
        self,
        dataset: Dataset,
        neighbor: URIRef,
        predicate: URIRef,
        direction: str,
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> List[URIRef]:
        self.logger.info("[PLM][Neighbor] augmenting non-set")
        self.logger.info("    • neighbor → %s", neighbor)
        self.logger.info("    • predicate → %s", predicate)
        self.logger.info("    • direction → %s", direction)
        created: List[URIRef] = []
        if self._node_in_graph(dataset.knowledge_graph_source, neighbor):
            clone = self._clone_non_set_node(dataset.knowledge_graph_source, neighbor)
            created.append(clone)
            self._mirror_relation(dataset.knowledge_graph_source, src_aug, predicate, clone, direction)
        if self._node_in_graph(dataset.knowledge_graph_target, neighbor):
            clone = self._clone_non_set_node(dataset.knowledge_graph_target, neighbor)
            created.append(clone)
            self._mirror_relation(dataset.knowledge_graph_target, tgt_aug, predicate, clone, direction)
        return created

    def _mirror_relation(self, graph, subject: URIRef, predicate: URIRef, neighbor: URIRef, direction: str) -> None:
        if direction == "out":
            graph.add((subject, predicate, neighbor))
        else:
            graph.add((neighbor, predicate, subject))

    def _node_in_graph(self, graph, node: URIRef) -> bool:
        if not isinstance(node, URIRef):
            return False
        key = (graph.identifier, node)
        if key in self._membership_cache:
            return self._membership_cache[key]
        exists = self._has_triple(graph.triples((node, None, None))) or self._has_triple(
            graph.triples((None, None, node))
        )
        self._membership_cache[key] = exists
        return exists

    @staticmethod
    def _has_triple(iterator: Iterable) -> bool:
        try:
            next(iterator)
            return True
        except StopIteration:
            return False

    def _mint_augmented_uri(self, reference: URIRef) -> URIRef:
        self._id_counter += 1
        base = str(reference)
        return URIRef(f"{base}_aug{self._id_counter}")

    def _clone_non_set_node(self, graph, node: URIRef) -> URIRef:
        clone = self._mint_augmented_uri(node)
        graph.add((clone, self.derived_predicate, node))
        for _, predicate, obj in graph.triples((node, None, None)):
            if isinstance(obj, Literal):
                graph.add((clone, predicate, obj))
        self.logger.info("    • cloned → %s (graph=%s)", clone, graph.identifier)
        return clone

    def _compute_pair_budget(self, initial_pairs: int) -> Optional[int]:
        ratio_limit: Optional[int] = None
        if self.augmentation_ratio is not None:
            ratio_limit = max(1, math.ceil(initial_pairs * self.augmentation_ratio))

        config_limit = self.max_pairs_config
        if ratio_limit is None:
            return config_limit
        if config_limit is None:
            return ratio_limit
        return min(ratio_limit, config_limit)

    def _collect_node_triples(self, graph, node: URIRef) -> Dict[str, List[Dict[str, object]]]:
        def fmt(subject: URIRef, predicate: URIRef, obj) -> Dict[str, object]:
            is_literal = isinstance(obj, Literal)
            value = obj.toPython() if is_literal else str(obj)
            if not isinstance(value, str):
                value = str(value)
            return {
                "subject": str(subject),
                "predicate": str(predicate),
                "object": value,
                "object_is_literal": is_literal,
            }

        outgoing = [fmt(node, predicate, obj) for _, predicate, obj in graph.triples((node, None, None))]
        incoming = [fmt(subject, predicate, node) for subject, predicate, _ in graph.triples((None, None, node))]
        return {"outgoing": outgoing, "incoming": incoming}

    def _log_node_connections(self, graph: SetKnowledgeGraph, node: URIRef) -> None:
        outgoing: list[str] = []
        incoming: list[str] = []
        for direction, predicate, neighbor in self._iter_neighbors(graph, node):
            if isinstance(neighbor, Literal):
                value = neighbor.toPython()
                if not isinstance(value, str):
                    value = str(value)
                neighbor_repr = f'"{value}"'
            else:
                neighbor_repr = str(neighbor)
            if direction == "out":
                outgoing.append(f"{predicate} -> {neighbor_repr}")
            else:
                incoming.append(f"{predicate} <- {neighbor_repr}")

        self.logger.debug("[PLM][Node] %s", node)
        if outgoing:
            self.logger.debug("    • outgoing (%d)", len(outgoing))
            for rel in outgoing:
                self.logger.debug("        -> %s", rel)
        if incoming:
            self.logger.debug("    • incoming (%d)", len(incoming))
            for rel in incoming:
                self.logger.debug("        <- %s", rel)

    # ------------------------------------------------------------------
    # Placeholder for PLM expansion
    # ------------------------------------------------------------------
    def _expansion(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        set_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        """Placeholder for PLM-driven literal interpolation (implemented later)."""
        self.logger.debug(
            "Expansion step skipped for node %s ->",
            set_node
        )

        self.logger.debug("          • %s",
            src_aug
        )

        self.logger.debug("          • %s",
            tgt_aug
        )
