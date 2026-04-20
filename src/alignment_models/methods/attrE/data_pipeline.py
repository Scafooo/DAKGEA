"""Data pipeline: Dataset → AttrEDataBundle.

Converts the project-standard Dataset object (two RDFlib-backed KnowledgeGraphs
+ aligned entity pairs) into the internal data structures required by the
AttrE PyTorch model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]
# (subj_id, pred_id, obj_id)
RelTriple = Tuple[int, int, int]
# (subj_id, pred_id, char_seq, weight)   char_seq: List[int] of length CHAR_SEQ_LEN
AttrTriple = Tuple[int, int, List[int], float]

CHAR_SEQ_LEN: int = 10  # characters per literal value (truncated / padded)


@dataclass
class AttrEDataBundle:
    """All data consumed by AttrETrainer."""

    # Entity vocabulary
    entity2id: Dict[str, int]
    id2entity: Dict[int, str]

    # Predicate vocabulary
    pred2id: Dict[str, int]
    id2pred: Dict[int, str]

    # Character vocabulary (built from literal values)
    char2id: Dict[str, int]
    id2char: Dict[int, str]

    # KG1 / KG2 entity ID sets (used for evaluation filtering)
    kg1_entity_ids: List[int]
    kg2_entity_ids: List[int]

    # Triples (integer IDs)
    rel_triples_1: List[RelTriple]
    rel_triples_2: List[RelTriple]
    attr_triples_1: List[AttrTriple]
    attr_triples_2: List[AttrTriple]

    # Alignment splits
    train_pairs: List[Pair]   # (kg1_ent_id, kg2_ent_id) used as supervision
    test_pairs: List[Pair]    # held out for evaluation

    # Negative-sampling pools
    neg_pool_1: List[int]   # KG1 entity IDs for negative sampling
    neg_pool_2: List[int]   # KG2 entity IDs for negative sampling

    char_seq_len: int = CHAR_SEQ_LEN

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_predicates(self) -> int:
        return len(self.pred2id)

    @property
    def num_chars(self) -> int:
        return len(self.char2id)


def build_attre_data(
    dataset: Dataset,
    train_ratio: float = 0.3,
    char_seq_len: int = CHAR_SEQ_LEN,
) -> AttrEDataBundle:
    """Convert *dataset* into an :class:`AttrEDataBundle`.

    Args:
        dataset: Project-standard Dataset (two KGs + aligned entities).
        train_ratio: Fraction of aligned pairs used for training supervision.
        char_seq_len: Number of characters kept per literal value.

    Returns:
        Fully populated AttrEDataBundle ready for AttrETrainer.
    """
    kg1 = dataset.knowledge_graph_source
    kg2 = dataset.knowledge_graph_target

    # ------------------------------------------------------------------
    # 1. Build unified entity + predicate vocabularies
    # ------------------------------------------------------------------
    all_entities: List[str] = []
    all_preds: List[str] = []

    kg1_entity_uris: set = set()
    kg2_entity_uris: set = set()

    for s, p, o in kg1:
        all_preds.append(str(p))
        if isinstance(s, URIRef):
            kg1_entity_uris.add(str(s))
        if isinstance(o, URIRef):
            kg1_entity_uris.add(str(o))

    for s, p, o in kg2:
        all_preds.append(str(p))
        if isinstance(s, URIRef):
            kg2_entity_uris.add(str(s))
        if isinstance(o, URIRef):
            kg2_entity_uris.add(str(o))

    # Entities from alignment that might not appear in any triple
    for e1, e2 in dataset.aligned_entities:
        kg1_entity_uris.add(str(e1))
        kg2_entity_uris.add(str(e2))

    all_entity_uris = sorted(kg1_entity_uris | kg2_entity_uris)
    entity2id = {e: i for i, e in enumerate(all_entity_uris)}
    id2entity = {i: e for e, i in entity2id.items()}

    unique_preds = sorted(set(all_preds))
    pred2id = {p: i for i, p in enumerate(unique_preds)}
    id2pred = {i: p for p, i in pred2id.items()}

    kg1_entity_ids = [entity2id[e] for e in sorted(kg1_entity_uris)]
    kg2_entity_ids = [entity2id[e] for e in sorted(kg2_entity_uris)]

    logger.info(
        "[AttrE] Vocab: %d entities (%d KG1, %d KG2), %d predicates",
        len(entity2id), len(kg1_entity_ids), len(kg2_entity_ids), len(pred2id),
    )

    # ------------------------------------------------------------------
    # 1b. Build predicate remap from attribute_matches
    #
    # In cross-schema datasets (e.g. DBpedia–Wikidata) the same attribute
    # (e.g. birth date) has different predicate URIs in the two KGs.
    # AttrE's alignment signal only works when aligned entities share the
    # *same* predicate embedding.  dataset.attribute_matches maps
    #   KG1_pred_uri -> [KG2_pred_uri, ...]
    # We remap KG2 predicate IDs to their KG1 counterpart so both KGs use
    # the same embedding for matched predicates.
    # ------------------------------------------------------------------
    attr_matches: Dict[str, List[str]] = getattr(dataset, "attribute_matches", None) or {}
    pred_remap: Dict[int, int] = {}   # kg2_pred_id → kg1_pred_id
    if attr_matches:
        n_remapped = 0
        for kg1_pred, kg2_preds in attr_matches.items():
            if kg1_pred not in pred2id:
                continue
            kg1_pid = pred2id[kg1_pred]
            for kg2_pred in kg2_preds:
                if kg2_pred in pred2id:
                    kg2_pid = pred2id[kg2_pred]
                    if kg2_pid != kg1_pid:
                        pred_remap[kg2_pid] = kg1_pid
                        n_remapped += 1
        logger.info(
            "[AttrE] Predicate remap: %d KG2 predicates → KG1 equivalents "
            "(from %d attribute_matches entries)",
            n_remapped, len(attr_matches),
        )
    else:
        logger.warning(
            "[AttrE] No attribute_matches found — KG1 and KG2 predicates are "
            "treated independently. Alignment signal may be weak on cross-schema datasets."
        )

    # ------------------------------------------------------------------
    # 2. Build character vocabulary from literal values
    # ------------------------------------------------------------------
    char2id: Dict[str, int] = {"<PAD>": 0}

    for kg in (kg1, kg2):
        for s, p, o in kg:
            if isinstance(o, Literal):
                for ch in str(o)[:char_seq_len]:
                    if ch not in char2id:
                        char2id[ch] = len(char2id)

    # Also add characters from entity labels (local name)
    for uri in all_entity_uris:
        label = _label_from_uri(uri)
        for ch in label[:char_seq_len]:
            if ch not in char2id:
                char2id[ch] = len(char2id)

    id2char = {i: c for c, i in char2id.items()}
    logger.info("[AttrE] Character vocab size: %d", len(char2id))

    # ------------------------------------------------------------------
    # 3. Extract relation and attribute triples
    # ------------------------------------------------------------------
    def _pred_weight(freq: int, total: int) -> float:
        """IDF-based predicate importance weight, in (0, 1]."""
        if total == 0 or freq == 0:
            return 1.0
        return float(math.log(1.0 + total / freq) / math.log(1.0 + total))

    def _str_to_char_seq(text: str) -> List[int]:
        # Strip leading/trailing whitespace: KG2 raw values often have a
        # leading space from the TSV format ("\tpred\t value"), which would
        # break the char match against KG1 values that have no such space.
        text = text.strip()
        seq = [char2id.get(ch, 0) for ch in text[:char_seq_len]]
        seq += [0] * (char_seq_len - len(seq))   # pad to fixed length
        return seq

    def _extract_triples(
        kg,
        entity_uris_set,
        remap: Optional[Dict[int, int]] = None,
    ):
        """Extract relation and attribute triples from *kg*.

        Args:
            remap: Optional mapping ``{original_pred_id: canonical_pred_id}``.
                   Applied to both relation and attribute triples so that
                   matched predicates from different KGs share the same
                   embedding.
        """
        rel_triples: List[RelTriple] = []
        attr_triples_raw: List[Tuple[int, int, str]] = []  # (s, p, literal_str)
        pred_freq: Dict[int, int] = {}

        for s, p, o in kg:
            if not isinstance(s, URIRef):
                continue
            s_str, p_str = str(s), str(p)
            if s_str not in entity2id or p_str not in pred2id:
                continue
            s_id = entity2id[s_str]
            p_id = pred2id[p_str]
            if remap:
                p_id = remap.get(p_id, p_id)

            if isinstance(o, URIRef):
                o_str = str(o)
                if o_str in entity2id:
                    rel_triples.append((s_id, p_id, entity2id[o_str]))
            elif isinstance(o, Literal):
                attr_triples_raw.append((s_id, p_id, str(o)))
                pred_freq[p_id] = pred_freq.get(p_id, 0) + 1

        total_attr = len(attr_triples_raw)
        attr_triples: List[AttrTriple] = [
            (s_id, p_id, _str_to_char_seq(val), _pred_weight(pred_freq[p_id], total_attr))
            for s_id, p_id, val in attr_triples_raw
        ]
        return rel_triples, attr_triples

    rel_triples_1, attr_triples_1 = _extract_triples(kg1, kg1_entity_uris)
    rel_triples_2, attr_triples_2 = _extract_triples(kg2, kg2_entity_uris, remap=pred_remap)

    logger.info(
        "[AttrE] Triples — rel1=%d rel2=%d attr1=%d attr2=%d",
        len(rel_triples_1), len(rel_triples_2), len(attr_triples_1), len(attr_triples_2),
    )

    # ------------------------------------------------------------------
    # 4. Alignment train / test split
    # ------------------------------------------------------------------
    # dataset.aligned_entities contains ONLY training pairs when the pipeline
    # uses ForgetLabelsReducer (which separates train/test and stores the test
    # pool in dataset.fixed_test_pairs).  We must NOT filter all_pairs against
    # fixed_test_pairs — they are disjoint by construction.
    train_pairs = [(entity2id[str(e1)], entity2id[str(e2)]) for e1, e2 in dataset.aligned_entities]

    if getattr(dataset, "fixed_test_pairs", None) is not None:
        # Test pairs come directly from the fixed split, not from aligned_entities.
        # Also ensure their entities are in the vocab (they appear in triples, but
        # add them explicitly for safety).
        for e1, e2 in dataset.fixed_test_pairs:
            s1, s2 = str(e1), str(e2)
            if s1 not in entity2id:
                new_id = len(entity2id)
                entity2id[s1] = new_id
                id2entity[new_id] = s1
                kg1_entity_uris.add(s1)
                kg1_entity_ids.append(new_id)
            if s2 not in entity2id:
                new_id = len(entity2id)
                entity2id[s2] = new_id
                id2entity[new_id] = s2
                kg2_entity_uris.add(s2)
                kg2_entity_ids.append(new_id)

        test_pairs = [
            (entity2id[str(e1)], entity2id[str(e2)])
            for e1, e2 in dataset.fixed_test_pairs
            if str(e1) in entity2id and str(e2) in entity2id
        ]
    else:
        rng = random.Random(42)
        shuffled = list(train_pairs)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_ratio))
        train_pairs = shuffled[:n_train]
        test_pairs = shuffled[n_train:]

    if not test_pairs:
        logger.warning(
            "[AttrE] test_pairs is EMPTY — evaluation will return 0. "
            "fixed_test_pairs=%s, aligned_entities=%d",
            "set" if getattr(dataset, "fixed_test_pairs", None) is not None else "None",
            len(dataset.aligned_entities),
        )

    logger.info("[AttrE] Alignment split: train=%d test=%d", len(train_pairs), len(test_pairs))

    return AttrEDataBundle(
        entity2id=entity2id,
        id2entity=id2entity,
        pred2id=pred2id,
        id2pred=id2pred,
        char2id=char2id,
        id2char=id2char,
        kg1_entity_ids=kg1_entity_ids,
        kg2_entity_ids=kg2_entity_ids,
        rel_triples_1=rel_triples_1,
        rel_triples_2=rel_triples_2,
        attr_triples_1=attr_triples_1,
        attr_triples_2=attr_triples_2,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        neg_pool_1=list(kg1_entity_ids),
        neg_pool_2=list(kg2_entity_ids),
        char_seq_len=char_seq_len,
    )


def _label_from_uri(uri: str) -> str:
    """Extract a human-readable label from a URI (local name)."""
    frag = uri.split("/")[-1].split("#")[-1].split(":")[-1]
    return frag.replace("_", " ").strip() or uri
