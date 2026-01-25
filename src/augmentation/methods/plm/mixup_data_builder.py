"""Training data builder for Mix-up BART focused on Balanced Creativity.

Tasks included:
1. Denoising: noise(A) -> A (Stability)
2. Robust Translation: noise(A) -> B (Creativity + Cross-KG)
3. Identity: A -> A (Latent Anchoring)
"""

from __future__ import annotations

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
    if not text or len(text) < 3: return text
    chars = list(text)
    if len(chars) > 2:
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    if random.random() < 0.15 and len(chars) > 4:
        del chars[random.randint(0, len(chars) - 1)]
    return "".join(chars)

class MixupDataBuilder:
    def __init__(self, confidence_threshold: float = 0.6):
        self.matcher_service = AttributeMatchingService({
            "predicate_matching": {
                "use_value_similarity": True,
                "similarity_threshold": confidence_threshold,
                "name_weight": 0.8,
                "value_weight": 0.2
            }
        })

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def build_training_data(
        self,
        dataset: Dataset,
        noise_fn: Callable[[str], str] = _default_noise,
        max_pairs_per_pred: int = 5000,
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        logger.info("[MixupBuilder] Computing correspondences for MAXIMUM QUALITY training...")
        correspondences = self.matcher_service.compute_correspondences(dataset)
        
        canonical_map = {}
        for corr in correspondences:
            c_token = f"<{_local_name(corr.src_uri).upper()}>"
            canonical_map[str(corr.src_uri)] = c_token
            canonical_map[str(corr.tgt_uri)] = c_token
            
        kg_src = dataset.knowledge_graph_source
        kg_tgt = dataset.knowledge_graph_target
        rows = []

        aligned_pairs = list(dataset.aligned_entities)
        random.shuffle(aligned_pairs)

        # Predicate counter to ensure balance
        pred_counts = defaultdict(int)

        for s_uri, t_uri in aligned_pairs:
            s_lits = defaultdict(list)
            for _, p, o in kg_src.triples((s_uri, None, None)):
                if isinstance(o, Literal): s_lits[str(p)].append(str(o))
            
            t_lits = defaultdict(list)
            for _, p, o in kg_tgt.triples((t_uri, None, None)):
                if isinstance(o, Literal): t_lits[str(p)].append(str(o))

            for corr in correspondences:
                s_p, t_p = str(corr.src_uri), str(corr.tgt_uri)
                if s_p in s_lits and t_p in t_lits:
                    # Check predicate balance limit
                    p_tok = canonical_map[s_p]
                    if pred_counts[p_tok] >= max_pairs_per_pred:
                        continue

                    v_s_list, v_t_list = s_lits[s_p], t_lits[t_p]
                    
                    if len(v_s_list) >= len(v_t_list):
                        for vs in v_s_list:
                            vt = max(v_t_list, key=lambda x: self._string_similarity(vs, x))
                            self._add_balanced_tasks(rows, p_tok, vs, vt, noise_fn)
                    else:
                        for vt in v_t_list:
                            vs = max(v_s_list, key=lambda x: self._string_similarity(vt, x))
                            self._add_balanced_tasks(rows, p_tok, vs, vt, noise_fn)
                    
                    pred_counts[p_tok] += 1
            
            # REMOVED HARD LIMIT: if len(rows) > 120000: break

        return rows, canonical_map

    def _add_balanced_tasks(self, rows, p_tok, vs, vt, noise_fn):
        """Implementazione bilanciata con Identity e Cross-KG Robust Translation."""
        # 1. Denoising Identity (Stabilità entro il KG)
        # noise(A) -> A
        rows.append({"input": f"{p_tok} {noise_fn(vs)}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {noise_fn(vt)}", "target": f"{p_tok} {vt}"})
        
        # 2. Robust Cross-KG Translation (Creatività)
        # noise(A) -> B pulito
        rows.append({"input": f"{p_tok} {noise_fn(vs)}", "target": f"{p_tok} {vt}"})
        rows.append({"input": f"{p_tok} {noise_fn(vt)}", "target": f"{p_tok} {vs}"})
        
        # 3. Identity pura (Ancoraggio centri di massa)
        # A -> A
        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vt}"})