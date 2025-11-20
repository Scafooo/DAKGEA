"""Service for graph-based expansion and entity generation.

This service encapsulates the BFS expansion logic and entity generation,
extracted from PLMAugmenter for better separation of concerns.
"""

import math
import random
import logging
from collections import deque
from typing import Optional, List, Deque
from pathlib import Path

from rdflib import URIRef, Literal

from src.core.dataset import Dataset
from src.logger import get_logger

from ..models import ExpansionContext
from ..expansion_node import ExpansionNode
from ..neighbor_handler import NeighborHandler
from ..node_expander import NodeExpander
from ..set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from .bart_service import BARTService
from .attribute_matching_service import AttributeMatchingService

logger = get_logger(__name__)


class GraphExpansionService:
    """Service for expanding knowledge graphs through BFS traversal.

    This service coordinates:
    - BFS exploration of the Set Knowledge Graph
    - Entity creation and augmentation
    - Neighbor processing and relation creation
    """

    def __init__(
        self,
        bart_service: BARTService,
        matching_service: AttributeMatchingService,
        config: dict,
    ):
        """Initialize the graph expansion service.

        Args:
            bart_service: BART service for value interpolation
            matching_service: Attribute matching service
            config: Full configuration dictionary
        """
        self.bart_service = bart_service
        self.matching_service = matching_service
        self.config = config

        # Extract configuration
        augmentation_cfg = config.get("augmentation", config)
        experiment_cfg = config.get("experiment", config)

        # Expansion parameters
        self.max_depth = int(augmentation_cfg.get("max_depth", 1))
        self.augmentation_ratio = self._parse_ratio(augmentation_cfg.get("ratio"))
        self.max_pairs_config = self._parse_max_pairs(augmentation_cfg.get("max_pairs"))
        self.seed = int(experiment_cfg.get("seed", config.get("seed", 0)))

        # Provenance tracking
        self.add_derived_predicate = bool(augmentation_cfg.get("add_derived_predicate", False))
        derived_pred = augmentation_cfg.get("derived_predicate")
        self.derived_predicate = (
            URIRef(derived_pred) if derived_pred
            else URIRef("http://dakgea.org/augmentation/derivedFrom")
        )

        # Initialize helpers
        self.neighbor_handler = NeighborHandler()

        # Node expander will be initialized with BART interpolator
        self.node_expander: Optional[NodeExpander] = None

        logger.info(
            f"[GraphExpansion] Initialized (max_depth={self.max_depth}, "
            f"ratio={self.augmentation_ratio}, seed={self.seed})"
        )

    def _parse_ratio(self, ratio) -> Optional[float]:
        """Parse augmentation ratio from config."""
        if ratio is None:
            return None
        try:
            ratio_value = float(ratio)
            return ratio_value if ratio_value > 0 else None
        except (TypeError, ValueError):
            logger.warning(f"Invalid ratio '{ratio}', ignoring")
            return None

    def _parse_max_pairs(self, max_pairs) -> Optional[int]:
        """Parse max_pairs from config."""
        if max_pairs is None:
            return None
        try:
            return max(1, int(max_pairs))
        except (TypeError, ValueError):
            logger.warning(f"Invalid max_pairs '{max_pairs}', ignoring")
            return None

    def initialize_node_expander(self) -> None:
        """Initialize the node expander with BART interpolator."""
        if self.node_expander is not None:
            return  # Already initialized

        bart_cfg = self.config.get("bart", {})

        # Get the underlying interpolator for backward compatibility
        bart_interpolator = self.bart_service.trainer.get_interpolator()

        # Get alignment cache if available
        alignment_cache = None
        if hasattr(self.matching_service, 'cache'):
            alignment_cache = self.matching_service.cache

        self.node_expander = NodeExpander(
            self.derived_predicate,
            self.add_derived_predicate,
            bart_interpolator,
            self.config.get("predicate_matching", {}),
            alignment_cache,
            advanced_training_config=bart_cfg.get("advanced_training", {}),
            bart_config=bart_cfg,
        )

        logger.info("[GraphExpansion] Node expander initialized")

    def expand_dataset(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
    ) -> Dataset:
        """Expand the dataset by generating synthetic entity pairs.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph (aligned entities merged)

        Returns:
            Augmented dataset
        """
        # Ensure node expander is ready
        self.initialize_node_expander()

        initial_pairs = len(dataset.aligned_entities)
        if initial_pairs == 0:
            logger.warning("[GraphExpansion] No aligned entities, skipping expansion")
            return dataset

        # Get set nodes
        set_nodes = sorted(set_graph.iter_set_nodes(), key=lambda uri: str(uri))
        if not set_nodes:
            logger.warning("[GraphExpansion] No set nodes found, skipping expansion")
            return dataset

        # Calculate budget
        pair_budget = self._compute_pair_budget(initial_pairs)
        self._log_expansion_start(initial_pairs, pair_budget)

        # Randomize starting order
        rng = random.Random(self.seed)
        rng.shuffle(set_nodes)

        # Perform BFS expansion
        logger.info("[GraphExpansion] Starting BFS expansion...")
        expanded_pairs = self._bfs_expansion(dataset, set_graph, set_nodes, pair_budget)

        logger.info(
            f"[GraphExpansion] Expansion complete: {expanded_pairs} pairs generated "
            f"(target: {pair_budget if pair_budget else 'unbounded'})"
        )

        return dataset

    def _compute_pair_budget(self, initial_pairs: int) -> Optional[int]:
        """Compute the maximum number of pairs to generate."""
        ratio_limit = None
        if self.augmentation_ratio is not None:
            ratio_limit = max(1, math.ceil(initial_pairs * self.augmentation_ratio))

        config_limit = self.max_pairs_config

        if ratio_limit is None:
            return config_limit
        if config_limit is None:
            return ratio_limit

        return min(ratio_limit, config_limit)

    def _bfs_expansion(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        set_nodes: List[URIRef],
        pair_budget: Optional[int],
    ) -> int:
        """Perform BFS expansion of set nodes.

        This is the core expansion logic extracted from PLMAugmenter.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            set_nodes: List of set nodes to potentially expand
            pair_budget: Maximum pairs to create

        Returns:
            Number of pairs actually expanded
        """
        # Initialize expansion context
        context = ExpansionContext(remaining_seeds=deque(set_nodes))

        while context.has_remaining_work(pair_budget):
            # Refill queue from seeds if empty
            context.refill_queue_from_seeds()

            if not context.queue:
                break

            # Process next node
            exp_node = context.queue.popleft()

            if exp_node.uri in context.visited:
                continue

            context.mark_visited(exp_node.uri)

            # Expand based on node type
            if exp_node.is_set_node:
                self._expand_set_node(
                    dataset, set_graph, exp_node, context, pair_budget
                )
            else:
                self._expand_non_set_node(
                    dataset, set_graph, exp_node, context
                )

        # Log summary
        self._log_expansion_summary(
            context.expanded_pairs, pair_budget, context.expansion_chain
        )

        return context.expanded_pairs

    def _expand_set_node(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        exp_node: ExpansionNode,
        context: ExpansionContext,
        pair_budget: Optional[int],
    ) -> None:
        """Expand a set node and process its neighbors."""
        set_node = exp_node.uri
        depth = exp_node.depth

        # Create augmented pair
        src_aug, tgt_aug, src_original, tgt_original = self.node_expander.expand_set_node(
            dataset, set_graph, set_node
        )

        # Record augmentation
        context.record_set_augmentation(set_node, src_aug, tgt_aug)

        # Bootstrap literal attributes
        self.node_expander.bootstrap_literals(
            dataset, src_original, tgt_original, src_aug, tgt_aug
        )

        # Process neighbors if within depth limit
        if depth < self.max_depth:
            self._process_neighbors(
                dataset, set_graph, set_node, src_aug, tgt_aug,
                depth, context, pair_budget
            )

    def _expand_non_set_node(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        exp_node: ExpansionNode,
        context: ExpansionContext,
    ) -> None:
        """Expand a non-set node (intermediate node in BFS)."""
        non_set_node = exp_node.uri

        # Get or create augmented version
        if non_set_node not in context.non_set_to_augmented:
            logger.debug(f"[GraphExpansion] Expanding non-set node from BFS: {non_set_node}")

            # Expand with self-interpolation
            src_aug, tgt_aug = self._expand_non_set_node_impl(dataset, set_graph, non_set_node)
            context.record_non_set_augmentation(non_set_node, src_aug, tgt_aug)

            # Process neighbors (if needed for further expansion)
            # This is handled by _expand_non_set_node_with_neighbors if we need full neighbor processing
            logger.info(f"[GraphExpansion][NonSet] Expanded non-set node from BFS: {non_set_node}")

    def _process_neighbors(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        current_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        current_depth: int,
        context: ExpansionContext,
        pair_budget: Optional[int],
    ) -> None:
        """Process all neighbors of current node."""
        direct_enqueued = set()

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

            # Handle set node neighbors
            if set_graph.is_set_node(neighbor):
                self._handle_set_neighbor(
                    dataset, set_graph, neighbor, predicate, direction,
                    current_node, src_aug, tgt_aug, current_depth,
                    context, direct_enqueued
                )
                continue

            # Handle non-set node neighbors
            self._handle_non_set_neighbor(
                dataset, set_graph, neighbor, predicate, direction,
                current_node, src_aug, tgt_aug, current_depth,
                context
            )

    def _handle_set_neighbor(
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
        context: ExpansionContext,
        direct_enqueued: set,
    ) -> None:
        """Handle a set node neighbor."""
        # If neighbor was already expanded, create relations
        augmentation = context.get_set_augmentation(neighbor)
        if augmentation:
            neighbor_src_aug, neighbor_tgt_aug = augmentation
            self._create_set_to_set_relations(
                dataset, set_graph, current_node, neighbor,
                src_aug, tgt_aug, neighbor_src_aug, neighbor_tgt_aug,
                predicate, direction
            )

        # Enqueue for expansion if not visited
        next_depth = current_depth + 1
        if (
            next_depth <= self.max_depth
            and neighbor not in context.visited
            and neighbor not in direct_enqueued
        ):
            context.queue.append(
                ExpansionNode(uri=neighbor, depth=next_depth, node_type="set", parent=current_node)
            )
            direct_enqueued.add(neighbor)

    def _create_set_to_set_relations(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        current_node: URIRef,
        neighbor_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        neighbor_src_aug: URIRef,
        neighbor_tgt_aug: URIRef,
        predicate: URIRef,
        direction: str,
    ) -> None:
        """Create relations between augmented entities.

        Only creates relations in graphs where they originally existed.
        """
        # Get relation origins
        origins = set_graph.get_relation_origins(current_node)
        relation_key = (predicate, neighbor_node, direction)

        exists_in_source = relation_key in origins.get("source", set())
        exists_in_target = relation_key in origins.get("target", set())

        # Create relations based on direction
        if direction == "out":
            if exists_in_source:
                dataset.knowledge_graph_source.add((src_aug, predicate, neighbor_src_aug))
            if exists_in_target:
                dataset.knowledge_graph_target.add((tgt_aug, predicate, neighbor_tgt_aug))
        else:
            if exists_in_source:
                dataset.knowledge_graph_source.add((neighbor_src_aug, predicate, src_aug))
            if exists_in_target:
                dataset.knowledge_graph_target.add((neighbor_tgt_aug, predicate, tgt_aug))

    def _log_expansion_start(self, initial_pairs: int, pair_budget: Optional[int]) -> None:
        """Log expansion initialization."""
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        logger.info(
            f"[GraphExpansion] Starting expansion: budget={budget_display}, "
            f"initial_pairs={initial_pairs}, ratio={self.augmentation_ratio}, "
            f"max_depth={self.max_depth}, seed={self.seed}"
        )

    def _log_expansion_summary(
        self, expanded_pairs: int, pair_budget: Optional[int], expansion_chain: List[str]
    ) -> None:
        """Log expansion summary."""
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        logger.info(
            f"[GraphExpansion] Expanded {expanded_pairs}/{budget_display} pairs "
            f"(max_depth={self.max_depth})"
        )

        if expansion_chain:
            chain_repr = " -> ".join(expansion_chain[:10])
            if len(expansion_chain) > 10:
                chain_repr += f" ... (+{len(expansion_chain) - 10} more)"
            logger.info(f"[GraphExpansion] Expansion chain: {chain_repr}")

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
        context: ExpansionContext,
    ) -> None:
        """Handle a non-set node neighbor.

        Non-set nodes can be:
        1. Expanded with self-interpolation if depth allows
        2. Connected to augmented entities
        3. Used for bridging to other set nodes
        """
        next_depth = current_depth + 1
        should_expand = next_depth < self.max_depth and neighbor not in context.non_set_to_augmented

        if should_expand:
            # Expand the non-set node with self-interpolation
            neighbor_src_aug, neighbor_tgt_aug = self._expand_non_set_node_impl(
                dataset, set_graph, neighbor
            )
            context.record_non_set_augmentation(neighbor, neighbor_src_aug, neighbor_tgt_aug)

            # Connect to the current augmented entities
            self._connect_non_set_to_set(
                dataset, neighbor, neighbor_src_aug, neighbor_tgt_aug,
                predicate, direction, src_aug, tgt_aug
            )

            logger.info(f"[GraphExpansion][NonSet] Expanded non-set node: {neighbor}")

            # Enqueue for further exploration if depth allows
            if next_depth < self.max_depth:
                context.queue.append(
                    ExpansionNode(uri=neighbor, depth=next_depth, node_type="non_set", parent=current_node)
                )
                context.mark_visited(neighbor)
        else:
            # Just connect without expanding (depth limit reached or already expanded)
            if neighbor in context.non_set_to_augmented:
                # Already expanded, use augmented version
                neighbor_src_aug, neighbor_tgt_aug = context.get_non_set_augmentation(neighbor)
                self._connect_non_set_to_set(
                    dataset, neighbor, neighbor_src_aug, neighbor_tgt_aug,
                    predicate, direction, src_aug, tgt_aug
                )
            else:
                # Connect original non-set node to augmented entities
                self.neighbor_handler.connect_non_set_to_augmented(
                    dataset, neighbor, predicate, direction, src_aug, tgt_aug
                )

        # Check for bridging to other set nodes (2-hop expansion)
        if current_depth + 2 <= self.max_depth:
            self._check_bridging(
                set_graph, neighbor, current_node, current_depth, context
            )

    def _expand_non_set_node_impl(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        non_set_node: URIRef,
    ) -> tuple[URIRef, URIRef]:
        """Expand a non-set node by creating augmented versions with self-interpolation.

        This generates variations of the non-set node's attributes using self-interpolation
        (similar to unmatched attributes in set nodes).
        """
        # Generate unique augmented URIs
        aug_counter = getattr(self, '_non_set_aug_counter', 0)
        self._non_set_aug_counter = aug_counter + 1

        # Determine which graph(s) contain this node
        in_source = self.neighbor_handler._node_in_graph(dataset.knowledge_graph_source, non_set_node)
        in_target = self.neighbor_handler._node_in_graph(dataset.knowledge_graph_target, non_set_node)

        if in_source:
            src_aug = URIRef(f"{non_set_node}_aug{aug_counter}")
        else:
            src_aug = None

        if in_target:
            tgt_aug = URIRef(f"{non_set_node}_aug{aug_counter + 1}")
        else:
            tgt_aug = None

        # Expand attributes from source graph
        if in_source and src_aug:
            literals = list(dataset.knowledge_graph_source.predicate_objects(non_set_node))
            for predicate, obj in literals:
                if isinstance(obj, Literal):
                    # Self-interpolate the literal (pass same value twice to BART)
                    try:
                        val = str(obj)
                        generated, _ = self.node_expander.bart_interpolator.interpolate_pair(
                            val, val, predicate=str(predicate)
                        )
                        if generated:
                            dataset.knowledge_graph_source.add((src_aug, predicate, Literal(generated)))
                            logger.debug(
                                f"[GraphExpansion] [src] non-set {predicate}: '{val[:30]}' → '{generated[:30]}'"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[GraphExpansion] [src] Failed to generate for {predicate}: {e}"
                        )

        # Expand attributes from target graph
        if in_target and tgt_aug:
            literals = list(dataset.knowledge_graph_target.predicate_objects(non_set_node))
            for predicate, obj in literals:
                if isinstance(obj, Literal):
                    # Self-interpolate the literal (pass same value twice to BART)
                    try:
                        val = str(obj)
                        generated, _ = self.node_expander.bart_interpolator.interpolate_pair(
                            val, val, predicate=str(predicate)
                        )
                        if generated:
                            dataset.knowledge_graph_target.add((tgt_aug, predicate, Literal(generated)))
                            logger.debug(
                                f"[GraphExpansion] [tgt] non-set {predicate}: '{val[:30]}' → '{generated[:30]}'"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[GraphExpansion] [tgt] Failed to generate for {predicate}: {e}"
                        )

        return src_aug, tgt_aug

    def _connect_non_set_to_set(
        self,
        dataset: Dataset,
        neighbor: URIRef,
        neighbor_src_aug: URIRef,
        neighbor_tgt_aug: URIRef,
        predicate: URIRef,
        direction: str,
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        """Connect non-set augmented node to set augmented node."""
        if direction == "out":
            # current --predicate--> neighbor
            if neighbor_src_aug and self.neighbor_handler._node_in_graph(dataset.knowledge_graph_source, neighbor):
                dataset.knowledge_graph_source.add((src_aug, predicate, neighbor_src_aug))
            if neighbor_tgt_aug and self.neighbor_handler._node_in_graph(dataset.knowledge_graph_target, neighbor):
                dataset.knowledge_graph_target.add((tgt_aug, predicate, neighbor_tgt_aug))
        else:
            # neighbor --predicate--> current
            if neighbor_src_aug and self.neighbor_handler._node_in_graph(dataset.knowledge_graph_source, neighbor):
                dataset.knowledge_graph_source.add((neighbor_src_aug, predicate, src_aug))
            if neighbor_tgt_aug and self.neighbor_handler._node_in_graph(dataset.knowledge_graph_target, neighbor):
                dataset.knowledge_graph_target.add((neighbor_tgt_aug, predicate, tgt_aug))

    def _check_bridging(
        self,
        set_graph: SetKnowledgeGraph,
        intermediate_node: URIRef,
        source_set_node: URIRef,
        current_depth: int,
        context: ExpansionContext,
    ) -> None:
        """Check for 2-hop paths to other set nodes (bridging).

        This allows expansion through non-set intermediary nodes.
        """
        # Iterate neighbors of the intermediate node
        for direction, predicate, neighbor in self.neighbor_handler.iter_neighbors(
            set_graph, intermediate_node
        ):
            # Skip non-URI neighbors
            if not isinstance(neighbor, URIRef):
                continue

            # Check if neighbor is a set node
            if set_graph.is_set_node(neighbor):
                # Found a 2-hop path: source_set_node -> intermediate_node -> neighbor (set node)
                # Enqueue the set node if not visited
                if neighbor not in context.visited and neighbor != source_set_node:
                    next_depth = current_depth + 2  # 2-hop distance
                    if next_depth <= self.max_depth:
                        context.queue.append(
                            ExpansionNode(
                                uri=neighbor,
                                depth=next_depth,
                                node_type="set",
                                parent=source_set_node
                            )
                        )
                        logger.debug(
                            f"[GraphExpansion][Bridge] Found 2-hop path: "
                            f"{source_set_node} -> {intermediate_node} -> {neighbor}"
                        )
