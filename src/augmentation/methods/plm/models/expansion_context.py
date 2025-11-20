"""Domain model for graph expansion context."""

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Deque
from collections import deque
from rdflib import URIRef

from ..expansion_node import ExpansionNode


@dataclass
class ExpansionContext:
    """Encapsulates the state during BFS graph expansion.

    This centralizes all the state that was previously scattered across
    the PLMAugmenter's _bfs_expansion method.
    """

    # BFS queue
    queue: Deque[ExpansionNode] = field(default_factory=deque)

    # Remaining seed nodes to explore
    remaining_seeds: Deque[URIRef] = field(default_factory=deque)

    # Visited nodes
    visited: Set[URIRef] = field(default_factory=set)

    # Mapping: set_node -> (augmented_src, augmented_tgt)
    set_node_to_augmented: Dict[URIRef, Tuple[URIRef, URIRef]] = field(
        default_factory=dict
    )

    # Mapping: non_set_node -> (augmented_src, augmented_tgt)
    non_set_to_augmented: Dict[URIRef, Tuple[URIRef, URIRef]] = field(
        default_factory=dict
    )

    # Expansion chain for logging
    expansion_chain: list = field(default_factory=list)

    # Counter for expanded pairs
    expanded_pairs: int = 0

    def has_remaining_work(self, pair_budget: int = None) -> bool:
        """Check if there's more work to do."""
        within_budget = pair_budget is None or self.expanded_pairs < pair_budget
        has_nodes = bool(self.queue or self.remaining_seeds)
        return within_budget and has_nodes

    def mark_visited(self, uri: URIRef) -> None:
        """Mark a node as visited."""
        self.visited.add(uri)
        self.expansion_chain.append(str(uri))

    def record_set_augmentation(
        self, set_node: URIRef, src_aug: URIRef, tgt_aug: URIRef
    ) -> None:
        """Record a set node augmentation."""
        self.set_node_to_augmented[set_node] = (src_aug, tgt_aug)
        self.expanded_pairs += 1

    def record_non_set_augmentation(
        self, non_set_node: URIRef, src_aug: URIRef, tgt_aug: URIRef
    ) -> None:
        """Record a non-set node augmentation."""
        self.non_set_to_augmented[non_set_node] = (src_aug, tgt_aug)

    def get_set_augmentation(
        self, set_node: URIRef
    ) -> Tuple[URIRef, URIRef] | None:
        """Get augmented entities for a set node."""
        return self.set_node_to_augmented.get(set_node)

    def get_non_set_augmentation(
        self, non_set_node: URIRef
    ) -> Tuple[URIRef, URIRef] | None:
        """Get augmented entities for a non-set node."""
        return self.non_set_to_augmented.get(non_set_node)

    def refill_queue_from_seeds(self) -> None:
        """Refill the queue from remaining seeds if empty."""
        if not self.queue and self.remaining_seeds:
            # Remove already visited seeds
            while self.remaining_seeds and self.remaining_seeds[0] in self.visited:
                self.remaining_seeds.popleft()

            if self.remaining_seeds:
                next_seed = self.remaining_seeds.popleft()
                self.queue.append(ExpansionNode(uri=next_seed, depth=0, node_type="set"))
