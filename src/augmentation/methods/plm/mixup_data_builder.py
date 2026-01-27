"""Training data builder for Mix-up BART with Special Tokens and Masked Reconstruction."""

import random
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from rdflib import Literal
from src.core.dataset import Dataset
from src.augmentation.methods.plm.services.attribute_matching_service import AttributeMatchingService

logger = logging.getLogger(__name__)

def _local_name(uri: str) -> str:
    return str(uri).split("/")[-1].split("#")[-1].upper()

class SpecialTokenNoiseGenerator:
    """Rumore strutturale: masking e shuffling per stimolare la creatività."""
    @staticmethod
    def apply(text: str) -> str:
        if not text or len(text) < 5: return text
        words = text.split()
        if len(words) < 2: return text
        
        strategy = random.random()
        # 1. Masking (40%) - Forza il modello a 'indovinare' i pezzi mancanti
        if strategy < 0.4:
            idx = random.randint(0, len(words)-1)
            words[idx] = "..." 
            return " ".join(words)
        # 2. Shuffling (30%)
        if strategy < 0.7:
            random.shuffle(words); return " ".join(words)
        # 3. Typo (30%)
        chars = list(text)
        i = random.randint(0, len(chars)-2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        return "".join(chars)

class MixupDataBuilder:
    def __init__(self, confidence_threshold: float = 0.6, value_match_threshold: float = 0.3):
        self.matcher_service = AttributeMatchingService({"predicate_matching": {"similarity_threshold": confidence_threshold}})
        self.value_match_threshold = value_match_threshold
        self.noise_gen = SpecialTokenNoiseGenerator()

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def build_training_data(self, dataset: Dataset, max_pairs_per_pred: int = 5000) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        logger.info("[MixupBuilder] Building Special-Token training data for BART...")
        correspondences = self.matcher_service.compute_correspondences(dataset)
        
        # Mappa URI -> Special Token (<NAME>, <DATE>, etc)
        canonical_map = {str(p): f"<{_local_name(p)}>" for p in (set(dataset.knowledge_graph_source.predicates()) | set(dataset.knowledge_graph_target.predicates()))}

        kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
        rows = []
        pred_counts = defaultdict(int)

        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): [str(o) for o in kg_src.objects(s_uri, p) if isinstance(o, Literal)] for p in kg_src.predicates(s_uri)}
            t_lits = {str(p): [str(o) for o in kg_tgt.objects(t_uri, p) if isinstance(o, Literal)] for p in kg_tgt.predicates(t_uri)}

            for corr in correspondences:
                sp, tp = str(corr.src_uri), str(corr.tgt_uri)
                if sp in s_lits and tp in t_lits and t_lits[tp]:
                    p_tok = canonical_map[sp]
                    if pred_counts[p_tok] >= max_pairs_per_pred: continue
                    
                    for vs in s_lits[sp]:
                        best_vt = max(t_lits[tp], key=lambda x: self._string_similarity(vs, x))
                        if self._string_similarity(vs, best_vt) >= self.value_match_threshold:
                            # ALLINEAMENTO BIDIREZIONALE (A <-> B)
                            rows.append({"input": f"{p_tok} {self.noise_gen.apply(vs)}", "target": f"{p_tok} {best_vt}"})
                            rows.append({"input": f"{p_tok} {self.noise_gen.apply(best_vt)}", "target": f"{p_tok} {vs}"})
                            
                            # IDENTITY LIMITATA
                            if random.random() < 0.5:
                                rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
                            
                            pred_counts[p_tok] += 1

            # ORFANI
            for p, vals in s_lits.items():
                p_tok = canonical_map.get(p)
                if p_tok and p not in canonical_map: # Solo veri orfani
                    for v in vals: rows.append({"input": f"{p_tok} {self.noise_gen.apply(v)}", "target": f"{p_tok} {v}"})

        return rows, canonical_map