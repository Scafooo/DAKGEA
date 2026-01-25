"""Training data builder for Mix-up BART with Orphan Attribute Learning and Coherent Pairing."""

import random
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
from difflib import SequenceMatcher
from rdflib import Literal
from src.core.dataset import Dataset
from src.augmentation.methods.plm.services.attribute_matching_service import AttributeMatchingService

logger = logging.getLogger(__name__)

def _local_name(uri: str) -> str:
    return str(uri).split("/")[-1].split("#")[-1]

def _default_noise(text: str) -> str:
    if not text or len(text) < 5: return text
    chars = list(text)
    if random.random() < 0.15:
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)

class MixupDataBuilder:
    def __init__(self, confidence_threshold: float = 0.6, value_match_threshold: float = 0.3):
        self.matcher_service = AttributeMatchingService({"predicate_matching": {"similarity_threshold": confidence_threshold}})
        self.value_match_threshold = value_match_threshold

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def build_training_data(self, dataset: Dataset, noise_fn: Callable[[str], str] = _default_noise, max_pairs_per_pred: int = 5000) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        logger.info("[MixupBuilder] Computing coherent correspondences...")
        correspondences = self.matcher_service.compute_correspondences(dataset)
        
        canonical_map = {}
        all_preds = set(dataset.knowledge_graph_source.predicates()) | set(dataset.knowledge_graph_target.predicates())
        for p in all_preds:
            canonical_map[str(p)] = f"<{_local_name(p).upper()}>"

        kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
        rows = []
        pred_counts = defaultdict(int)

        for s_uri, t_uri in dataset.aligned_entities:
            used_s, used_t = set(), set()
            s_lits = {str(p): [str(o) for o in kg_src.objects(s_uri, p) if isinstance(o, Literal)] for p in kg_src.predicates(s_uri)}
            t_lits = {str(p): [str(o) for o in kg_tgt.objects(t_uri, p) if isinstance(o, Literal)] for p in kg_tgt.predicates(t_uri)}

            for corr in correspondences:
                sp, tp = str(corr.src_uri), str(corr.tgt_uri)
                if sp in s_lits and tp in t_lits:
                    p_tok = canonical_map[sp]
                    if pred_counts[p_tok] >= max_pairs_per_pred: continue
                    
                    # COHERENT PAIRING: Trova il miglior match tra i valori
                    for vs in s_lits[sp]:
                        best_vt = max(t_lits[tp], key=lambda x: self._string_similarity(vs, x))
                        sim = self._string_similarity(vs, best_vt)
                        
                        if sim >= self.value_match_threshold:
                            # ALLINEAMENTO COERENTE (es. "The Beatles" + "Beatles, The")
                            self._add_balanced_tasks(rows, p_tok, vs, best_vt, noise_fn)
                            used_s.add(sp); used_t.add(tp)
                            pred_counts[p_tok] += 1
                        else:
                            # TROPPO DIVERSI: Trattali come orfani separati (es. "Nome" + "Data")
                            rows.append({"input": f"{p_tok} {noise_fn(vs)}", "target": f"{p_tok} {vs}"})
                            rows.append({"input": f"{canonical_map[tp]} {noise_fn(best_vt)}", "target": f"{canonical_map[tp]} {best_vt}"})

            # ORFANI PURI (predicati non mappati)
            for p, vals in s_lits.items():
                if p not in used_s:
                    for v in vals: rows.append({"input": f"{canonical_map[p]} {noise_fn(v)}", "target": f"{canonical_map[p]} {v}"})
            for p, vals in t_lits.items():
                if p not in used_t:
                    for v in vals: rows.append({"input": f"{canonical_map[p]} {noise_fn(v)}", "target": f"{canonical_map[p]} {v}"})

        return rows, canonical_map

    def _add_balanced_tasks(self, rows, p_tok, vs, vt, noise_fn):
        # noise(A) -> B, noise(B) -> A, A -> A, B -> B
        rows.append({"input": f"{p_tok} {noise_fn(vs)}", "target": f"{p_tok} {vt}"})
        rows.append({"input": f"{p_tok} {noise_fn(vt)}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vt}"})
