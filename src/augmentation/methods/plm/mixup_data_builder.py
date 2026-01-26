"""Training data builder for Mix-up BART with Diverse Value Pairing.

Key insight: The original PLM generated "interesting names" because it created
DIVERSE pairs by cross-producting all values. The similarity-based pairing
eliminates this diversity, causing the model to only learn identity mappings.

Solution: Use GLOBAL DIVERSE PAIRING - collect all values for each predicate
across all entities and create cross-pairs between DIFFERENT values.
"""

import random
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher
from rdflib import Literal
from src.core.dataset import Dataset
from src.augmentation.methods.plm.services.attribute_matching_service import AttributeMatchingService

logger = logging.getLogger(__name__)

def _local_name(uri: str) -> str:
    return str(uri).split("/")[-1].split("#")[-1]

def _basic_noise(text: str) -> str:
    """Apply light character-level noise for DAE training."""
    if not text or len(text) < 5: return text
    chars = list(text)
    if random.random() < 0.15:
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


class MixupDataBuilder:
    """Builds training data for Mix-up BART using diverse value pairing.

    The key innovation is GLOBAL DIVERSE PAIRING: instead of pairing each value
    with its most similar counterpart (which produces copies), we collect all
    values for each predicate and create cross-pairs between DIFFERENT values.

    This teaches the model to TRANSFORM values (e.g., "John Smith" → "Jane Doe"),
    not just copy them. During inference, mix-up in latent space between diverse
    embeddings produces genuinely new outputs.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        value_match_threshold: float = 0.3,
        diverse_pairs_per_value: int = 3,  # How many diverse pairs per value
        min_diversity_threshold: float = 0.7,  # Max similarity for "diverse" pairs
    ):
        self.matcher_service = AttributeMatchingService({
            "predicate_matching": {"similarity_threshold": confidence_threshold}
        })
        self.value_match_threshold = value_match_threshold
        self.diverse_pairs_per_value = diverse_pairs_per_value
        self.min_diversity_threshold = min_diversity_threshold

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _sample_diverse_partners(self, value: str, all_values: List[str], n: int) -> List[str]:
        """Sample n diverse partners for a value (preferring dissimilar values).

        This is the key function that ensures diversity in training pairs.
        Instead of picking the most similar value, we pick values that are
        DIFFERENT enough to teach the model real transformations.
        """
        if len(all_values) <= 1:
            return []

        # Calculate similarity to all other values
        candidates = []
        for v in all_values:
            if v.lower().strip() == value.lower().strip():
                continue  # Skip identical values
            sim = self._string_similarity(value, v)
            # Prefer values that are different but not completely random
            # Sweet spot: 0.2 < similarity < 0.7
            if sim < self.min_diversity_threshold:
                candidates.append((v, sim))

        if not candidates:
            # Fallback: use any different value
            candidates = [(v, self._string_similarity(value, v))
                         for v in all_values
                         if v.lower().strip() != value.lower().strip()]

        if not candidates:
            return []

        # Sort by similarity (ascending) to prefer more diverse pairs
        # But add some randomness to avoid always picking the same pairs
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[1])

        # Take the most diverse ones
        selected = [c[0] for c in candidates[:n]]
        return selected

    def build_training_data(
        self,
        dataset: Dataset,
        max_pairs_per_pred: int = 5000
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Build training data with diverse pairing strategy.

        Training tasks:
        1. Identity (A → A): Anchors the latent space
        2. DAE (noise(A) → A): Robustness to noise
        3. Diverse Cross (A → B where A ≠ B): Teaches real transformations

        The diverse cross-pairs are the key innovation that enables the model
        to generate creative outputs instead of copies.
        """
        logger.info("[MixupBuilder] Computing coherent correspondences...")
        correspondences = self.matcher_service.compute_correspondences(dataset)

        # Build canonical map for all predicates
        all_predicates = (set(dataset.knowledge_graph_source.predicates()) |
                         set(dataset.knowledge_graph_target.predicates()))
        canonical_map = {str(p): f"<{_local_name(p).upper()}>" for p in all_predicates}

        kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target

        # PHASE 1: Collect global value pools per canonical predicate
        # This enables diverse pairing across different entities
        global_pools: Dict[str, Set[str]] = defaultdict(set)

        rows = []
        pred_counts = defaultdict(int)

        # PHASE 2: Process aligned entities for identity + DAE tasks
        # Also populate global pools
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): [str(o) for o in kg_src.objects(s_uri, p) if isinstance(o, Literal)]
                     for p in kg_src.predicates(s_uri)}
            t_lits = {str(p): [str(o) for o in kg_tgt.objects(t_uri, p) if isinstance(o, Literal)]
                     for p in kg_tgt.predicates(t_uri)}

            # Add to global pools and create identity/DAE tasks
            for corr in correspondences:
                sp, tp = str(corr.src_uri), str(corr.tgt_uri)
                if sp in s_lits:
                    p_tok = canonical_map[sp]
                    for vs in s_lits[sp]:
                        global_pools[p_tok].add(vs)
                        # Identity task
                        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
                        # DAE task
                        rows.append({"input": f"{p_tok} {_basic_noise(vs)}", "target": f"{p_tok} {vs}"})

                if tp in t_lits:
                    p_tok_t = canonical_map[tp]
                    for vt in t_lits[tp]:
                        global_pools[p_tok_t].add(vt)
                        # Identity task
                        rows.append({"input": f"{p_tok_t} {vt}", "target": f"{p_tok_t} {vt}"})
                        # DAE task
                        rows.append({"input": f"{p_tok_t} {_basic_noise(vt)}", "target": f"{p_tok_t} {vt}"})

            # Handle orphan predicates (not in correspondences)
            used_preds = {str(c.src_uri) for c in correspondences} | {str(c.tgt_uri) for c in correspondences}
            for p, vals in s_lits.items():
                if p not in used_preds:
                    p_tok = canonical_map[p]
                    for v in vals:
                        global_pools[p_tok].add(v)
                        rows.append({"input": f"{p_tok} {v}", "target": f"{p_tok} {v}"})
                        rows.append({"input": f"{p_tok} {_basic_noise(v)}", "target": f"{p_tok} {v}"})
            for p, vals in t_lits.items():
                if p not in used_preds:
                    p_tok = canonical_map[p]
                    for v in vals:
                        global_pools[p_tok].add(v)
                        rows.append({"input": f"{p_tok} {v}", "target": f"{p_tok} {v}"})
                        rows.append({"input": f"{p_tok} {_basic_noise(v)}", "target": f"{p_tok} {v}"})

        # PHASE 3: Create DIVERSE CROSS-PAIRS from global pools
        # This is the key innovation that teaches the model to transform values
        logger.info("[MixupBuilder] Creating diverse cross-pairs...")

        for p_tok, values in global_pools.items():
            if pred_counts[p_tok] >= max_pairs_per_pred:
                continue

            values_list = list(values)
            if len(values_list) < 2:
                continue  # Need at least 2 values to create pairs

            # For each value, create pairs with DIVERSE partners
            for v1 in values_list:
                if pred_counts[p_tok] >= max_pairs_per_pred:
                    break

                diverse_partners = self._sample_diverse_partners(
                    v1, values_list, self.diverse_pairs_per_value
                )

                for v2 in diverse_partners:
                    if pred_counts[p_tok] >= max_pairs_per_pred:
                        break

                    # Bidirectional cross-mapping
                    rows.append({"input": f"{p_tok} {v1}", "target": f"{p_tok} {v2}"})
                    rows.append({"input": f"{p_tok} {v2}", "target": f"{p_tok} {v1}"})
                    pred_counts[p_tok] += 1

        # Log statistics
        total_identity_dae = sum(1 for r in rows if r["input"].split(" ", 1)[1] == r["target"].split(" ", 1)[1] or
                                  _basic_noise(r["target"].split(" ", 1)[1]) == r["input"].split(" ", 1)[1])
        total_cross = len(rows) - total_identity_dae
        logger.info(f"[MixupBuilder] Built {len(rows)} training samples:")
        logger.info(f"  - Identity/DAE: ~{total_identity_dae}")
        logger.info(f"  - Diverse Cross: ~{total_cross}")
        logger.info(f"  - Predicates with values: {len([p for p, v in global_pools.items() if len(v) > 0])}")

        return rows, canonical_map

    def _add_balanced_tasks(self, rows, p_tok, vs, vt):
        """Add balanced training tasks for aligned value pairs.

        Note: This method is kept for backward compatibility but the main
        build_training_data now uses the global diverse pairing strategy.
        """
        # 1-2. Identity: anchors latent space
        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vt}"})
        # 3-4. DAE: robustness to noise
        rows.append({"input": f"{p_tok} {_basic_noise(vs)}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {_basic_noise(vt)}", "target": f"{p_tok} {vt}"})
        # 5-6. Cross: semantic mapping (ALWAYS, even if similar)
        # The diverse pairing in Phase 3 handles the creative transformations
        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vt}"})
        rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vs}"})
