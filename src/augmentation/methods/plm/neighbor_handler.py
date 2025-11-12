"""Neighbor handling logic for PLM augmentation."""

from typing import Iterator, List, Tuple

from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger

from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph

logger = get_logger(__name__)


class NeighborHandler:
    """Handles neighbor processing during graph expansion."""

    @staticmethod
    def iter_neighbors(
        graph: SetKnowledgeGraph, node: URIRef
    ) -> Iterator[Tuple[str, URIRef, URIRef | Literal]]:
        """Iterate over all neighbors of a node (both incoming and outgoing).

        Args:
            graph: The set knowledge graph
            node: The node to get neighbors for

        Yields:
            Tuples of (direction, predicate, neighbor) where direction is "out" or "in"
        """
        # Outgoing edges
        for _, predicate, neighbor in graph.triples((node, None, None)):
            yield "out", predicate, neighbor

        # Incoming edges
        for neighbor, predicate, _ in graph.triples((None, None, node)):
            yield "in", predicate, neighbor

    def attach_literal_to_augmented(
        self,
        dataset: Dataset,
        predicate: URIRef,
        literal: Literal,
        direction: str,
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        """Attach a literal to augmented entities.

        Only attaches outgoing literals (direction="out") to maintain graph structure.

        Args:
            dataset: Dataset to modify
            predicate: Predicate URI
            literal: Literal value to attach
            direction: "out" or "in"
            src_aug: Augmented source entity
            tgt_aug: Augmented target entity
        """
        if direction != "out":
            return

        dataset.knowledge_graph_source.add((src_aug, predicate, literal))
        dataset.knowledge_graph_target.add((tgt_aug, predicate, literal))
        # Log removed - already covered by node_expander logging

    def find_bridged_set_neighbors(
        self,
        set_graph: SetKnowledgeGraph,
        non_set_node: URIRef,
        current_set_node: URIRef,
    ) -> List[URIRef]:
        """Find set nodes that are connected through a non-set node (bridging).

        This identifies set nodes that can be reached by traversing through a non-set node,
        enabling 2-hop expansion in the BFS.

        Args:
            set_graph: The set knowledge graph
            non_set_node: The intermediate non-set node
            current_set_node: The current set node (to avoid cycles)

        Returns:
            List of set node URIs that are bridged through the non-set node
        """
        bridged: List[URIRef] = []

        for direction, predicate, neighbor in self.iter_neighbors(set_graph, non_set_node):
            # Only consider URI neighbors
            if not isinstance(neighbor, URIRef):
                continue

            # Only consider set nodes
            if not set_graph.is_set_node(neighbor):
                continue

            # Avoid going back to the current node
            if neighbor == current_set_node:
                continue

            bridged.append(neighbor)
            logger.debug(
                "    • found bridged set node: %s --%s--> %s --%s--> %s",
                current_set_node,
                predicate,
                non_set_node,
                predicate,
                neighbor,
            )

        return bridged

    def connect_non_set_to_augmented(
        self,
        dataset: Dataset,
        non_set_node: URIRef,
        predicate: URIRef,
        direction: str,
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        """Connect a non-set node to augmented entities.

        This creates edges between the non-set node and the augmented entities,
        preserving the direction of the original connection.

        Args:
            dataset: Dataset to modify
            non_set_node: The non-set node to connect
            predicate: Predicate for the connection
            direction: "out" or "in" indicating edge direction
            src_aug: Augmented source entity
            tgt_aug: Augmented target entity
        """
        logger.info("[PLM][NonSet] Connecting non-set neighbor")
        logger.info("    • neighbor → %s", non_set_node)
        logger.info("    • predicate → %s", predicate)
        logger.info("    • direction → %s", direction)

        # Check if node exists in source/target graphs and connect accordingly
        if self._node_in_graph(dataset.knowledge_graph_source, non_set_node):
            self._mirror_relation(
                dataset.knowledge_graph_source, src_aug, predicate, non_set_node, direction
            )

        if self._node_in_graph(dataset.knowledge_graph_target, non_set_node):
            self._mirror_relation(
                dataset.knowledge_graph_target, tgt_aug, predicate, non_set_node, direction
            )

    @staticmethod
    def _mirror_relation(
        graph, subject: URIRef, predicate: URIRef, obj: URIRef, direction: str
    ) -> None:
        """Add a relation to the graph respecting the specified direction."""
        if direction == "out":
            graph.add((subject, predicate, obj))
        else:
            graph.add((obj, predicate, subject))

    @staticmethod
    def _node_in_graph(graph, node: URIRef) -> bool:
        """Check if a node exists in the graph (has any incoming or outgoing edges)."""
        if not isinstance(node, URIRef):
            return False

        # Check if node appears as subject or object in any triple
        try:
            next(graph.triples((node, None, None)))
            return True
        except StopIteration:
            pass

        try:
            next(graph.triples((None, None, node)))
            return True
        except StopIteration:
            return False
