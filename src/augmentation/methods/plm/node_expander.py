"""Node expansion logic for PLM augmentation."""

from typing import Optional, Tuple

from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger

from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph

logger = get_logger(__name__)


class NodeExpander:
    """Handles the expansion of nodes during PLM augmentation."""

    def __init__(self, derived_predicate: URIRef, add_derived_predicate: bool = False):
        """Initialize the node expander.

        Args:
            derived_predicate: Predicate URI to use for derivation tracking
            add_derived_predicate: Whether to actually add derivedFrom triples
        """
        self.derived_predicate = derived_predicate
        self.add_derived_predicate = add_derived_predicate
        self._id_counter = 0

    def expand_set_node(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        set_node: URIRef,
    ) -> Tuple[URIRef, URIRef, Optional[URIRef], Optional[URIRef]]:
        """Expand a set node by creating an aligned augmented pair.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            set_node: The set node to expand

        Returns:
            Tuple of (src_aug, tgt_aug, src_original, tgt_original)
        """
        # Extract source and target components from the set node
        components = set_graph.get_components(set_node)
        src_component = components[0] if components else None
        tgt_component = components[1] if len(components) > 1 else None

        # Create new augmented URIs
        src_aug = self._mint_augmented_uri(src_component or set_node)
        tgt_aug = self._mint_augmented_uri(tgt_component or set_node)

        # Optionally add derivation tracking
        if self.add_derived_predicate:
            src_reference = src_component or set_node
            tgt_reference = tgt_component or set_node
            dataset.knowledge_graph_source.add((src_aug, self.derived_predicate, src_reference))
            dataset.knowledge_graph_target.add((tgt_aug, self.derived_predicate, tgt_reference))

        # Add the new alignment
        alignments = list(dataset.aligned_entities)
        alignments.append((str(src_aug), str(tgt_aug)))
        dataset.aligned_entities = tuple(alignments)

        logger.info("[PLM] Expanded set node → %s", set_node)
        logger.info("    • src_aug → %s", src_aug)
        logger.info("    • tgt_aug → %s", tgt_aug)

        return src_aug, tgt_aug, src_component, tgt_component

    def bootstrap_literals(
        self,
        dataset: Dataset,
        src_component: Optional[URIRef],
        tgt_component: Optional[URIRef],
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        """Copy literal attributes from original entities to augmented ones.

        Args:
            dataset: Dataset containing the knowledge graphs
            src_component: Original source entity URI (or None)
            tgt_component: Original target entity URI (or None)
            src_aug: Augmented source entity URI
            tgt_aug: Augmented target entity URI
        """
        self._copy_literals(dataset.knowledge_graph_source, src_component, src_aug)
        self._copy_literals(dataset.knowledge_graph_target, tgt_component, tgt_aug)

    @staticmethod
    def _copy_literals(graph, original: Optional[URIRef], augmented: URIRef) -> None:
        """Copy all literal attributes from original to augmented entity."""
        if not original:
            return

        literal_count = 0
        for _, predicate, obj in graph.triples((original, None, None)):
            if isinstance(obj, Literal):
                graph.add((augmented, predicate, obj))
                literal_count += 1

        if literal_count > 0:
            logger.debug("    • bootstrapped %d literals", literal_count)

    def _mint_augmented_uri(self, reference: URIRef) -> URIRef:
        """Generate a new augmented URI based on a reference URI.

        Args:
            reference: The reference URI to base the new URI on

        Returns:
            A new unique augmented URI
        """
        self._id_counter += 1
        base = str(reference)
        return URIRef(f"{base}_aug{self._id_counter}")
