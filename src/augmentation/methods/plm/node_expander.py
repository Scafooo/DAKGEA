"""Node expansion logic for PLM augmentation."""
import logging
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
        predicate_matcher_config: Optional[dict] = None,
        predicate_alignment_cache=None,  # Pre-computed alignment cache
        advanced_training_config: Optional[dict] = None,  # Advanced training config
        bart_config: Optional[dict] = None,  # Full BART config for unmatched generation
    ):
        """Initialize the node expander.

        Args:
            derived_predicate: Predicate URI to use for derivation tracking
            add_derived_predicate: Whether to actually add derivedFrom triples
            bart_interpolator: Optional BART interpolator for literal generation
            predicate_matcher_config: Configuration for semantic predicate matching
            predicate_alignment_cache: Pre-computed PredicateAlignmentCache (optional)
            advanced_training_config: Advanced training configuration
            bart_config: Full BART configuration (for unmatched attributes generation)
        """
        self.derived_predicate = derived_predicate
        self.add_derived_predicate = add_derived_predicate
        self.bart_interpolator = bart_interpolator
        self._id_counter = 0

        # Initialize predicate matcher (lazy loading, used only if no cache)
        self.predicate_matcher = None
        self.predicate_matcher_config = predicate_matcher_config or {}

        # Pre-computed alignment cache (preferred method)
        self.alignment_cache = predicate_alignment_cache

        # Advanced training config (kept for backward compatibility)
        self.advanced_training_config = advanced_training_config or {}

        # Unmatched attributes generation config
        bart_cfg = bart_config or {}
        self.generate_unmatched = bart_cfg.get("generate_unmatched", False)
        self.unmatched_sample_rate = float(bart_cfg.get("unmatched_sample_rate", 0.5))

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

        logger.info("[PLM] Expanding set node: %s", set_node)
        logger.info("  Original → Source: %s | Target: %s", src_component or "None", tgt_component or "None")
        logger.info("  Augmented → Source: %s | Target: %s", src_aug, tgt_aug)

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

        Uses semantic predicate matching to find corresponding predicates
        between source and target, then generates interpolated literal values.
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

        # Log attributes of original entities
        src_preview = ", ".join(list(src_literals.keys())[:3]) + ("..." if len(src_literals) > 3 else "")
        tgt_preview = ", ".join(list(tgt_literals.keys())[:3]) + ("..." if len(tgt_literals) > 3 else "")
        logger.info("  Attributes: Source[%d]=%s | Target[%d]=%s",
                   len(src_literals), src_preview, len(tgt_literals), tgt_preview)

        # Try pre-computed alignment cache first (if available)
        matches = []
        if self.alignment_cache:
            logger.info("  Matching predicates using pre-computed cache...")
            matches = self._get_matches_from_cache(src_literals, tgt_literals)

            # If cache didn't find any matches, fallback to on-the-fly matching
            if not matches:
                logger.info("  No matches in cache, fallback to on-the-fly matching")
                # Fallback to on-the-fly matching
                if self.predicate_matcher is None:
                    from .predicate_matcher import PredicateMatcher
                    logger.info("  └─ Initializing semantic predicate matcher")
                    self.predicate_matcher = PredicateMatcher(self.predicate_matcher_config)
                    logger.info(f"  └─ Model: {self.predicate_matcher.embedding_model_name}")
                    logger.info(f"  └─ Similarity threshold: {self.predicate_matcher.similarity_threshold}")

                # Get attr_names from knowledge graphs
                src_attr_names = dataset.knowledge_graph_source.attr_to_name
                tgt_attr_names = dataset.knowledge_graph_target.attr_to_name

                # Find matching predicates using semantic similarity
                matches = self.predicate_matcher.match_predicates(
                    src_literals, tgt_literals, src_attr_names, tgt_attr_names
                )
        else:
            # No cache available, use on-the-fly matching directly
            # Initialize predicate matcher if needed (lazy loading)
            if self.predicate_matcher is None:
                from .predicate_matcher import PredicateMatcher
                logger.info("Matching Predicates:")
                logger.info("  └─ Initializing semantic predicate matcher")
                self.predicate_matcher = PredicateMatcher(self.predicate_matcher_config)
                logger.info(f"  └─ Model: {self.predicate_matcher.embedding_model_name}")
                logger.info(f"  └─ Similarity threshold: {self.predicate_matcher.similarity_threshold}")

            # Get attr_names from knowledge graphs
            src_attr_names = dataset.knowledge_graph_source.attr_to_name
            tgt_attr_names = dataset.knowledge_graph_target.attr_to_name

            # Find matching predicates using semantic similarity
            matches = self.predicate_matcher.match_predicates(
                src_literals, tgt_literals, src_attr_names, tgt_attr_names
            )

        if not matches:
            logger.info("  ⚠ No matching predicates found, all will be treated as unmatched")
            # Don't return - continue to generate unmatched attributes
            interpolated_count = 0
        else:
            # Deduplicate: keep only best match per source predicate
            best_matches = {}
            for match in matches:
                src = match.src_predicate
                if src not in best_matches or match.confidence > best_matches[src].confidence:
                    best_matches[src] = match

            matches = list(best_matches.values())
            matches.sort(key=lambda m: m.confidence, reverse=True)

            avg_confidence = sum(m.confidence for m in matches) / len(matches)
            logger.info("  ✓ Found %d best matches (avg confidence: %.3f)", len(matches), avg_confidence)
            interpolated_count = 0

        for idx, match in enumerate(matches, 1):
            src_pred, src_vals = src_literals[match.src_predicate]
            tgt_pred, tgt_vals = tgt_literals[match.tgt_predicate]

            # Use first value from each (or could sample randomly)
            src_val = src_vals[0] if src_vals else ""
            tgt_val = tgt_vals[0] if tgt_vals else ""

            if not src_val or not tgt_val:
                continue

            # Interpolate using BART
            try:
                aug_src_val, aug_tgt_val = self.bart_interpolator.interpolate_pair(
                    str(src_val),
                    str(tgt_val),
                    predicate=match.src_predicate,
                )

                # Add interpolated literals to augmented entities
                dataset.knowledge_graph_source.add(
                    (src_aug, src_pred, Literal(aug_src_val))
                )
                dataset.knowledge_graph_target.add(
                    (tgt_aug, tgt_pred, Literal(aug_tgt_val))
                )
                interpolated_count += 1

                # Log interpolation details with match confidence
                src_str = str(src_val)[:30]
                tgt_str = str(tgt_val)[:30]
                aug_src_str = str(aug_src_val)[:30]
                aug_tgt_str = str(aug_tgt_val)[:30]

                logger.info("    [%d/%d] %s ↔ %s (conf:%.2f) | '%s' + '%s' → '%s' / '%s'",
                           idx, len(matches), match.src_predicate, match.tgt_predicate, match.confidence,
                           src_str, tgt_str, aug_src_str, aug_tgt_str)

            except Exception as e:
                logger.warning("    [%d/%d] ✗ Failed %s ↔ %s: %s",
                              idx, len(matches), match.src_predicate, match.tgt_predicate, e)

        if interpolated_count > 0:
            logger.info("  ✓ Generated %d matched attributes", interpolated_count)

        # Generate variations for unmatched attributes (if enabled)
        unmatched_count = 0
        if self.generate_unmatched and self.bart_interpolator:
            unmatched_count = self._generate_unmatched_attributes(
                dataset, src_component, tgt_component, src_aug, tgt_aug,
                src_literals, tgt_literals, matches
            )

        # Recap: count total triples generated for this node
        # Matched generates triples for BOTH source and target (2x)
        # Unmatched generates triples only for the side that had the unmatched predicate (1x)
        matched_triples = interpolated_count * 2  # Both source and target
        unmatched_triples = unmatched_count  # Only one side per unmatched
        total_triples = matched_triples + unmatched_triples

        logger.info("")
        logger.info("  📊 RECAP for this node:")
        logger.info("    • Matched attributes: %d (→ %d triples, both sides)", interpolated_count, matched_triples)
        logger.info("    • Unmatched attributes: %d (→ %d triples, one side)", unmatched_count, unmatched_triples)
        logger.info("    • Total triples generated: %d", total_triples)
        logger.info("")

    def _generate_unmatched_attributes(
        self,
        dataset,
        src_component,
        tgt_component,
        src_aug,
        tgt_aug,
        src_literals: dict,
        tgt_literals: dict,
        matches: list,
    ):
        """Generate synthetic variations for attributes that didn't match.

        For attributes that exist only in source OR only in target (not in both),
        generate a small variation by passing the value twice to BART (self-interpolation).
        """
        import random

        # Get matched predicate names
        matched_src = {m.src_predicate for m in matches}
        matched_tgt = {m.tgt_predicate for m in matches}

        # Find unmatched predicates
        unmatched_src = [name for name in src_literals.keys() if name not in matched_src]
        unmatched_tgt = [name for name in tgt_literals.keys() if name not in matched_tgt]

        total_unmatched = len(unmatched_src) + len(unmatched_tgt)

        if total_unmatched == 0:
            logger.info("  All attributes matched, no unmatched to generate")
            return 0

        logger.info("  Unmatched: Source=%d | Target=%d | Total=%d",
                   len(unmatched_src), len(unmatched_tgt), total_unmatched)

        # Sample to avoid generating too many
        if self.unmatched_sample_rate < 1.0:
            num_src_sample = max(1, int(len(unmatched_src) * self.unmatched_sample_rate))
            num_tgt_sample = max(1, int(len(unmatched_tgt) * self.unmatched_sample_rate))
            unmatched_src = random.sample(unmatched_src, min(num_src_sample, len(unmatched_src)))
            unmatched_tgt = random.sample(unmatched_tgt, min(num_tgt_sample, len(unmatched_tgt)))
            logger.info("  Sampling %.0f%% → %d unmatched to generate",
                       self.unmatched_sample_rate * 100, len(unmatched_src) + len(unmatched_tgt))

        generated_count = 0

        # Generate for source unmatched attributes
        for pred_name in unmatched_src:
            pred_uri, values = src_literals[pred_name]
            if not values:
                continue

            val = str(values[0])

            try:
                aug_val, _ = self.bart_interpolator.interpolate_pair(val, val, predicate=pred_name)
                dataset.knowledge_graph_source.add((src_aug, pred_uri, Literal(aug_val)))
                generated_count += 1

                val_str = val[:30]
                aug_val_str = aug_val[:30]
                logger.info("    [src] %s | '%s' → '%s'", pred_name, val_str, aug_val_str)

            except Exception as e:
                logger.warning("    [src] ✗ %s: %s", pred_name, e)

        # Generate for target unmatched attributes
        for pred_name in unmatched_tgt:
            pred_uri, values = tgt_literals[pred_name]
            if not values:
                continue

            val = str(values[0])

            try:
                aug_val, _ = self.bart_interpolator.interpolate_pair(val, val, predicate=pred_name)
                dataset.knowledge_graph_target.add((tgt_aug, pred_uri, Literal(aug_val)))
                generated_count += 1

                val_str = val[:30]
                aug_val_str = aug_val[:30]
                logger.info("    [tgt] %s | '%s' → '%s'", pred_name, val_str, aug_val_str)

            except Exception as e:
                logger.warning("    [tgt] ✗ %s: %s", pred_name, e)

        if generated_count > 0:
            logger.info("  ✓ Generated %d unmatched attributes", generated_count)

        return generated_count

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

    def _get_matches_from_cache(
        self,
        src_literals: dict,
        tgt_literals: dict,
    ):
        """Get predicate matches from pre-computed alignment cache.

        Args:
            src_literals: Source predicates {local_name: (uri, [values])}
            tgt_literals: Target predicates {local_name: (uri, [values])}

        Returns:
            List of PredicateMatch objects (compatible with predicate_matcher output)
        """
        from .predicate_matcher import PredicateMatch

        matches = []

        # Debug logging (always show for now to debug the issue)
        logger.info(f"  Looking up matches in cache for {len(src_literals)} src × {len(tgt_literals)} tgt predicates")
        logger.info(f"    Source predicates: {list(src_literals.keys())}")
        logger.info(f"    Target predicates: {list(tgt_literals.keys())}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"    Source URIs: {[str(uri) for _, (uri, _) in src_literals.items()]}")
            logger.debug(f"    Target URIs: {[str(uri) for _, (uri, _) in tgt_literals.items()]}")

        for src_local, (src_uri, src_vals) in src_literals.items():
            for tgt_local, (tgt_uri, tgt_vals) in tgt_literals.items():
                # Look up in cache
                alignment = self.alignment_cache.get_alignment(src_uri, tgt_uri)

                if alignment:
                    # Check if values for THIS specific entity pair are similar enough
                    if src_vals and tgt_vals:
                        # Compute value similarity for this entity
                        value_sim = self._compute_value_similarity(src_vals, tgt_vals)

                        # Use a permissive threshold (0.3) for value filtering
                        if value_sim < 0.3:
                            logger.debug(f"    ✗ Filtered: {src_local} ↔ {tgt_local} (cache={alignment.combined_score:.3f}, "
                                       f"value_sim={value_sim:.3f} too low)")
                            continue

                        # Adjust confidence based on actual value similarity
                        adjusted_confidence = 0.7 * alignment.combined_score + 0.3 * value_sim
                    else:
                        adjusted_confidence = alignment.combined_score

                    # Convert to PredicateMatch for compatibility
                    match = PredicateMatch(
                        src_predicate=src_local,
                        tgt_predicate=tgt_local,
                        src_uri=src_uri,
                        tgt_uri=tgt_uri,
                        confidence=adjusted_confidence,
                        strategy="hybrid_alignment",
                    )
                    matches.append(match)
                    logger.info(f"    ✓ {src_local} ↔ {tgt_local} (score={adjusted_confidence:.3f})")
                elif src_local == tgt_local:  # Log when identical names don't match
                    logger.warning(f"    ✗ NOT in cache (SAME NAME!): {src_local}")
                    logger.warning(f"      Source URI: {src_uri}")
                    logger.warning(f"      Target URI: {tgt_uri}")

        # Sort by confidence (combined score)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        logger.info(f"  Total matches: {len(matches)}")

        return matches

    def _compute_value_similarity(self, src_vals: list, tgt_vals: list) -> float:
        """Compute similarity between literal values using character n-grams.

        Args:
            src_vals: List of source literal values
            tgt_vals: List of target literal values

        Returns:
            Jaccard similarity score [0, 1]
        """
        if not src_vals or not tgt_vals:
            return 0.0

        src_str = str(src_vals[0]).lower()
        tgt_str = str(tgt_vals[0]).lower()

        # Character 3-grams for fuzzy matching
        def get_ngrams(s: str, n: int = 3) -> set:
            s = " " + s + " "
            return {s[i:i+n] for i in range(len(s) - n + 1)}

        src_ngrams = get_ngrams(src_str)
        tgt_ngrams = get_ngrams(tgt_str)

        if not src_ngrams or not tgt_ngrams:
            return 0.0

        # Jaccard similarity
        intersection = len(src_ngrams & tgt_ngrams)
        union = len(src_ngrams | tgt_ngrams)

        return intersection / union if union > 0 else 0.0

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
