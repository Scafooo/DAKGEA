"""PLM-based augmentation strategy leveraging latent interpolation scaffolding.

Refactored version with improved modularity and clarity.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, List, Optional

from rdflib import Literal, URIRef

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset

from .expansion_node import ExpansionNode
from .neighbor_handler import NeighborHandler
from .node_expander import NodeExpander
from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via PLM-based expansion of fused entity sets.

    The augmentation process:
    1. Creates a SetKnowledgeGraph where aligned entities are fused
    2. Performs BFS expansion starting from random set nodes
    3. For each set node, creates an aligned pair of augmented entities
    4. Bootstraps literal attributes from original entities
    5. Expands to neighbors while maintaining structural coherence
    6. Handles bridging through non-set nodes for 2-hop expansion

    Configuration:
        - max_depth: Maximum BFS depth (default: 1)
        - ratio: Augmentation ratio (e.g., 0.5 = 50% more entities)
        - max_pairs: Absolute maximum number of pairs to generate
        - seed: Random seed for reproducibility
        - add_derived_predicate: Whether to add derivedFrom triples (default: False)
    """

    registry_name = "plm_augmentation"
    _DEFAULT_DERIVED_PREDICATE = URIRef("http://dakgea.org/augmentation/derivedFrom")

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        cfg = self.config or {}
        augmentation_cfg = cfg.get("augmentation", cfg)
        experiment_cfg = cfg.get("experiment", cfg)

        # Expansion parameters
        self.max_depth = int(augmentation_cfg.get("max_depth", 1))
        max_pairs = augmentation_cfg.get("max_pairs")
        self.max_pairs_config = max(1, int(max_pairs)) if max_pairs is not None else None

        # Augmentation ratio
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

        # Reproducibility
        seed = experiment_cfg.get("seed", cfg.get("seed", 0))
        self.seed = int(seed) if seed is not None else 0

        # Provenance tracking
        self.add_derived_predicate = bool(augmentation_cfg.get("add_derived_predicate", False))
        derived_predicate = augmentation_cfg.get("derived_predicate", cfg.get("derived_predicate"))
        self.derived_predicate = (
            URIRef(derived_predicate) if derived_predicate else self._DEFAULT_DERIVED_PREDICATE
        )

        # Initialize helper classes
        self.node_expander = NodeExpander(self.derived_predicate, self.add_derived_predicate)
        self.neighbor_handler = NeighborHandler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def augment(self, dataset: Dataset) -> Dataset:
        """Augment a dataset by spawning aligned synthetic entities.

        Args:
            dataset: Input dataset to augment

        Returns:
            Augmented dataset with new aligned entity pairs
        """
        initial_pairs = len(dataset.aligned_entities)
        dataset_augmented = dataset.clone()

        if not dataset.aligned_entities:
            self.logger.warning("Skipping PLM augmentation: no aligned entities available.")
            return dataset_augmented

        # Create fused set graph
        set_graph = SetKnowledgeGraph.from_dataset(dataset)
        set_nodes = sorted(set_graph.iter_set_nodes(), key=lambda uri: str(uri))

        if not set_nodes:
            self.logger.warning("SetKnowledgeGraph is empty; nothing to augment.")
            return dataset_augmented

        # Calculate expansion budget
        pair_budget = self._compute_pair_budget(initial_pairs)
        self._log_augmentation_start(initial_pairs, pair_budget)

        # Randomize starting order
        rng = random.Random(self.seed)
        rng.shuffle(set_nodes)

        # Perform BFS expansion
        self.section("PLM Augmentation")
        expanded_pairs = self._bfs_expansion(dataset_augmented, set_graph, set_nodes, pair_budget)

        self.logger.info(
            "[PLM] Augmentation complete: expanded %d pairs (target: %s)",
            expanded_pairs,
            pair_budget if pair_budget is not None else "unbounded",
        )

        return dataset_augmented

    # ------------------------------------------------------------------
    # BFS Expansion
    # ------------------------------------------------------------------
    def _bfs_expansion(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        set_nodes: List[URIRef],
        pair_budget: Optional[int],
    ) -> int:
        """Perform BFS expansion of set nodes.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            set_nodes: List of set nodes to potentially expand
            pair_budget: Maximum number of pairs to create (None = unlimited)

        Returns:
            Number of pairs actually expanded
        """
        visited: set[URIRef] = set()
        expansion_chain: List[str] = []
        queue: Deque[ExpansionNode] = deque()
        remaining_seeds: Deque[URIRef] = deque(set_nodes)
        expanded_pairs = 0

        while (queue or remaining_seeds) and (pair_budget is None or expanded_pairs < pair_budget):
            # Refill queue from remaining seeds if empty
            if not queue:
                while remaining_seeds and remaining_seeds[0] in visited:
                    remaining_seeds.popleft()
                if not remaining_seeds:
                    break
                next_seed = remaining_seeds.popleft()
                queue.append(ExpansionNode(uri=next_seed, depth=0, node_type="set"))

            # Process next node in queue
            exp_node = queue.popleft()

            if exp_node.uri in visited:
                continue

            visited.add(exp_node.uri)
            expansion_chain.append(str(exp_node.uri))

            # Expand the node
            if exp_node.is_set_node:
                self._expand_set_node_with_neighbors(
                    dataset, set_graph, exp_node, queue, visited
                )
                expanded_pairs += 1
            else:
                # Non-set nodes will be handled by PLM expansion (future work)
                self.logger.debug("[PLM] Non-set node encountered (future expansion): %s", exp_node.uri)

        # Log expansion summary
        self._log_expansion_summary(expanded_pairs, pair_budget, expansion_chain)

        return expanded_pairs

    def _expand_set_node_with_neighbors(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        exp_node: ExpansionNode,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
    ) -> None:
        """Expand a set node and process its neighbors.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            exp_node: Expansion node to process
            queue: BFS queue to add new nodes to
            visited: Set of already visited nodes
        """
        set_node = exp_node.uri
        depth = exp_node.depth

        # Log connections for debugging
        self._log_node_connections(set_graph, set_node)

        # Create augmented pair
        src_aug, tgt_aug, src_original, tgt_original = self.node_expander.expand_set_node(
            dataset, set_graph, set_node
        )

        # Bootstrap literal attributes
        self.node_expander.bootstrap_literals(
            dataset, src_original, tgt_original, src_aug, tgt_aug
        )

        # TODO: PLM-driven expansion (future implementation)
        # This will use a language model to generate variations of literals
        # self._plm_expansion(dataset, set_graph, set_node, src_aug, tgt_aug)

        # Process neighbors if we haven't reached max depth
        if depth < self.max_depth:
            self._process_neighbors(
                dataset, set_graph, set_node, src_aug, tgt_aug, depth, queue, visited
            )

    def _process_neighbors(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        current_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
    ) -> None:
        """Process all neighbors of the current node.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            current_node: Current set node being processed
            src_aug: Augmented source entity
            tgt_aug: Augmented target entity
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
        """
        direct_enqueued: set[URIRef] = set()

        for direction, predicate, neighbor in self.neighbor_handler.iter_neighbors(
            set_graph, current_node
        ):
            # Handle literal neighbors
            if isinstance(neighbor, Literal):
                self.neighbor_handler.attach_literal_to_augmented(
                    dataset, predicate, neighbor, direction, src_aug, tgt_aug
                )
                continue

            # Skip non-URI neighbors
            if not isinstance(neighbor, URIRef):
                continue

            # Handle set node neighbors (direct expansion)
            if set_graph.is_set_node(neighbor):
                self._enqueue_set_neighbor(
                    neighbor, current_node, current_depth, queue, visited, direct_enqueued
                )
                continue

            # Handle non-set node neighbors
            self._handle_non_set_neighbor(
                dataset,
                set_graph,
                neighbor,
                predicate,
                direction,
                current_node,
                src_aug,
                tgt_aug,
                current_depth,
                queue,
                visited,
            )

    def _enqueue_set_neighbor(
        self,
        neighbor: URIRef,
        current_node: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
        direct_enqueued: set[URIRef],
    ) -> None:
        """Enqueue a set node neighbor for direct expansion.

        Args:
            neighbor: The set node neighbor to enqueue
            current_node: Current node being processed
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
            direct_enqueued: Set of nodes already enqueued in this iteration
        """
        next_depth = current_depth + 1

        # Check if we should enqueue this neighbor
        if (
            next_depth <= self.max_depth
            and neighbor not in visited
            and neighbor not in direct_enqueued
        ):
            queue.append(ExpansionNode(uri=neighbor, depth=next_depth, node_type="set", parent=current_node))
            direct_enqueued.add(neighbor)

            self.logger.info("[PLM][Queue] Direct set neighbor enqueued")
            self.logger.info("    • current → %s", current_node)
            self.logger.info("    • neighbor → %s", neighbor)
            self.logger.info("    • depth → %d", next_depth)

    def _handle_non_set_neighbor(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        neighbor: URIRef,
        predicate: URIRef,
        direction: str,
        current_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
    ) -> None:
        """Handle a non-set node neighbor.

        Non-set nodes are connected to the augmented entities and checked for bridging
        to other set nodes.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            neighbor: The non-set neighbor node
            predicate: Predicate connecting to the neighbor
            direction: Direction of the edge ("out" or "in")
            current_node: Current set node
            src_aug: Augmented source entity
            tgt_aug: Augmented target entity
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
        """
        # Connect the non-set node to augmented entities
        self.neighbor_handler.connect_non_set_to_augmented(
            dataset, neighbor, predicate, direction, src_aug, tgt_aug
        )

        # Check for bridging to other set nodes (2-hop expansion)
        if current_depth + 2 <= self.max_depth:
            self._check_bridging(
                set_graph, neighbor, current_node, current_depth, queue, visited
            )

    def _check_bridging(
        self,
        set_graph: SetKnowledgeGraph,
        non_set_node: URIRef,
        current_node: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
    ) -> None:
        """Check if a non-set node bridges to other set nodes.

        Bridging allows 2-hop expansion: set_node -> non_set_node -> another_set_node

        Args:
            set_graph: Set knowledge graph
            non_set_node: The intermediate non-set node
            current_node: Current set node
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
        """
        bridged_nodes = self.neighbor_handler.find_bridged_set_neighbors(
            set_graph, non_set_node, current_node
        )

        bridged_enqueued: set[URIRef] = set()
        next_depth = current_depth + 2

        for bridged_node in bridged_nodes:
            if bridged_node in visited or bridged_node in bridged_enqueued:
                continue

            queue.append(
                ExpansionNode(
                    uri=bridged_node,
                    depth=next_depth,
                    node_type="set",
                    parent=non_set_node,
                )
            )
            bridged_enqueued.add(bridged_node)

            self.logger.info("[PLM][Queue] Bridged via non-set node")
            self.logger.info("    • current → %s", current_node)
            self.logger.info("    • bridge → %s", non_set_node)
            self.logger.info("    • next → %s", bridged_node)
            self.logger.info("    • depth → %d", next_depth)

    # ------------------------------------------------------------------
    # Budget Calculation
    # ------------------------------------------------------------------
    def _compute_pair_budget(self, initial_pairs: int) -> Optional[int]:
        """Compute the maximum number of pairs to generate.

        Takes the minimum of ratio-based and config-based limits.

        Args:
            initial_pairs: Number of aligned pairs in input dataset

        Returns:
            Maximum number of pairs to generate, or None for unlimited
        """
        ratio_limit: Optional[int] = None
        if self.augmentation_ratio is not None:
            ratio_limit = max(1, math.ceil(initial_pairs * self.augmentation_ratio))

        config_limit = self.max_pairs_config

        if ratio_limit is None:
            return config_limit
        if config_limit is None:
            return ratio_limit

        return min(ratio_limit, config_limit)

    # ------------------------------------------------------------------
    # Logging Helpers
    # ------------------------------------------------------------------
    def _log_augmentation_start(self, initial_pairs: int, pair_budget: Optional[int]) -> None:
        """Log augmentation initialization parameters."""
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        self.logger.info(
            "[PLM] Starting augmentation: budget=%s (initial=%d, ratio=%s, max_pairs=%s, seed=%d, derived=%s)",
            budget_display,
            initial_pairs,
            f"{self.augmentation_ratio:.2f}" if self.augmentation_ratio is not None else "-",
            self.max_pairs_config if self.max_pairs_config is not None else "-",
            self.seed,
            "enabled" if self.add_derived_predicate else "disabled",
        )

    def _log_expansion_summary(
        self, expanded_pairs: int, pair_budget: Optional[int], expansion_chain: List[str]
    ) -> None:
        """Log expansion summary."""
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        self.logger.info(
            "[PLM] Expanded %d/%s set nodes (max_depth=%d)",
            expanded_pairs,
            budget_display,
            self.max_depth,
        )

        if expansion_chain:
            chain_repr = " -> ".join(expansion_chain[:10])
            if len(expansion_chain) > 10:
                chain_repr += f" ... (+{len(expansion_chain) - 10} more)"
            self.logger.info("[PLM] Expansion chain: %s", chain_repr)
        else:
            self.logger.info("[PLM] Expansion chain: (none)")

    def _log_node_connections(self, graph: SetKnowledgeGraph, node: URIRef) -> None:
        """Log all connections of a node for debugging."""
        outgoing: list[str] = []
        incoming: list[str] = []

        for direction, predicate, neighbor in self.neighbor_handler.iter_neighbors(graph, node):
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
