"""Node expansion logic for PLM augmentation."""

from typing import Optional, Tuple, TYPE_CHECKING

from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger

from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph

if TYPE_CHECKING:
    from .bart_interpolator import BartInterpolatorPLM

logger = get_logger(__name__)


class NodeExpander:
    """Handles the expansion of nodes during PLM augmentation."""

    def __init__(
        self,
        derived_predicate: URIRef,
        add_derived_predicate: bool = False,
        bart_interpolator: Optional["BartInterpolatorPLM"] = None,
    ):
        """Initialize the node expander.

        Args:
            derived_predicate: Predicate URI to use for derivation tracking
            add_derived_predicate: Whether to actually add derivedFrom triples
            bart_interpolator: Optional BART interpolator for literal generation
        """
        self.derived_predicate = derived_predicate
        self.add_derived_predicate = add_derived_predicate
        self.bart_interpolator = bart_interpolator
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
        """Generate literal attributes for augmented entities using BART interpolation.

        Only generates literals if BART interpolator is available.
        Augmented entities without BART will have no literals.

        Args:
            dataset: Dataset containing the knowledge graphs
            src_component: Original source entity URI (or None)
            tgt_component: Original target entity URI (or None)
            src_aug: Augmented source entity URI
            tgt_aug: Augmented target entity URI
        """
        if self.bart_interpolator:
            self._interpolate_literals(
                dataset, src_component, tgt_component, src_aug, tgt_aug
            )
        else:
            logger.debug("    • no BART interpolator - skipping literals")

    def _interpolate_literals(
        self,
        dataset: Dataset,
        src_component: Optional[URIRef],
        tgt_component: Optional[URIRef],
        src_aug: URIRef,
        tgt_aug: URIRef,
    ) -> None:
        """Interpolate literals using BART for augmented entities.

        Collects matching predicates from source and target, then uses BART
        to generate interpolated literal values.
        """
        if not src_component or not tgt_component:
            logger.debug("    • missing components - skipping interpolation")
            return

        # Collect literals from both components, grouped by predicate local name
        src_literals = self._collect_predicate_literals(
            dataset.knowledge_graph_source, src_component
        )
        tgt_literals = self._collect_predicate_literals(
            dataset.knowledge_graph_target, tgt_component
        )

        # Find common predicates (by local name)
        common_predicates = set(src_literals.keys()) & set(tgt_literals.keys())

        interpolated_count = 0
        for pred_name in common_predicates:
            src_pred, src_vals = src_literals[pred_name]
            tgt_pred, tgt_vals = tgt_literals[pred_name]

            # Use first value from each (or could sample randomly)
            src_val = src_vals[0] if src_vals else ""
            tgt_val = tgt_vals[0] if tgt_vals else ""

            if not src_val or not tgt_val:
                continue

            # Interpolate using BART
            try:
                aug_src_val, aug_tgt_val = self.bart_interpolator.interpolate_pair(
                    str(src_val), str(tgt_val), predicate=pred_name
                )

                # Add interpolated literals to augmented entities
                dataset.knowledge_graph_source.add(
                    (src_aug, src_pred, Literal(aug_src_val))
                )
                dataset.knowledge_graph_target.add(
                    (tgt_aug, tgt_pred, Literal(aug_tgt_val))
                )
                interpolated_count += 1

                # Log interpolation details
                src_str = str(src_val)[:40]
                tgt_str = str(tgt_val)[:40]
                aug_src_str = str(aug_src_val)[:40]
                aug_tgt_str = str(aug_tgt_val)[:40]

                logger.info("      [%s]", pred_name)
                logger.info("        src: '%s' + '%s' → '%s'", src_str, tgt_str, aug_src_str)
                logger.info("        tgt: '%s' + '%s' → '%s'", src_str, tgt_str, aug_tgt_str)

            except Exception as e:
                logger.warning(
                    "Failed to interpolate predicate %s: %s. Skipping this predicate.",
                    pred_name, e
                )

        if interpolated_count > 0:
            logger.info("    • ✓ interpolated %d literals with BART", interpolated_count)

    @staticmethod
    def _collect_predicate_literals(graph, entity: URIRef) -> dict:
        """Collect literals grouped by predicate local name.

        Returns:
            Dict mapping predicate_local_name -> (predicate_uri, [literal_values])
        """
        from .bart_interpolator import _clean_pred

        pred_map = {}
        for _, predicate, obj in graph.triples((entity, None, None)):
            if isinstance(obj, Literal):
                local_name = _clean_pred(str(predicate))
                if local_name not in pred_map:
                    pred_map[local_name] = (predicate, [])
                pred_map[local_name][1].append(obj)
        return pred_map

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
