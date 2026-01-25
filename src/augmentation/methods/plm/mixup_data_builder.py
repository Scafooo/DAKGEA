"""Training data builder for Mix-up BART with Structural & Semantic Noise."""

import random
import logging
import re
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
from difflib import SequenceMatcher
from rdflib import Literal
from src.core.dataset import Dataset
from src.augmentation.methods.plm.services.attribute_matching_service import AttributeMatchingService

logger = logging.getLogger(__name__)

def _local_name(uri: str) -> str:
    return str(uri).split("/")[-1].split("#")[-1]

class RobustNoiseGenerator:
    """Generatore di rumore intelligente per stimolare la creatività del modello."""
    
    @staticmethod
    def apply(text: str) -> str:
        if not text or len(text) < 4: return text
        
        # Scegliamo una strategia di rumore
        strategy = random.random()
        words = text.split()
        
        # 1. Word Shuffling (30% chance) - Ottimo per nomi e titoli
        if strategy < 0.3 and len(words) >= 2:
            random.shuffle(words)
            return " ".join(words)
            
        # 2. Article/Stopword Dropping (20% chance)
        if strategy < 0.5:
            stops = {'the', 'a', 'an', 'of', 'in', 'and'}
            filtered = [w for w in words if w.lower() not in stops or random.random() > 0.7]
            if filtered: return " ".join(filtered)
            
        # 3. Partial Deletion (20% chance) - Forza la ricostruzione
        if strategy < 0.7 and len(words) >= 3:
            del words[random.randint(0, len(words)-1)]
            return " ".join(words)
            
        # 4. Classic Typo (20% chance)
        chars = list(text)
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

class MixupDataBuilder:
    def __init__(self, confidence_threshold: float = 0.6, value_match_threshold: float = 0.3):
        self.matcher_service = AttributeMatchingService({"predicate_matching": {"similarity_threshold": confidence_threshold}})
        self.value_match_threshold = value_match_threshold
        self.noise_gen = RobustNoiseGenerator()

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def build_training_data(self, dataset: Dataset, max_pairs_per_pred: int = 5000) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        logger.info("[MixupBuilder] Building training data with ROBUST NOISE...")
        correspondences = self.matcher_service.compute_correspondences(dataset)
        
        canonical_map = {str(p): f"<{_local_name(p).upper()}>" for p in (set(dataset.knowledge_graph_source.predicates()) | set(dataset.knowledge_graph_target.predicates()))}

        kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
        rows = []
        pred_counts = defaultdict(int)

        for s_uri, t_uri in dataset.aligned_entities:
            used_s, used_t = set(), set()
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
                            # ALLINEAMENTO (Training con rumore robusto)
                            self._add_balanced_tasks(rows, p_tok, vs, best_vt)
                            used_s.add(sp); used_t.add(tp)
                            pred_counts[p_tok] += 1
                        else:
                            # ORFANI DA DIVERSE SEMANTICS
                            rows.append({"input": f"{p_tok} {self.noise_gen.apply(vs)}", "target": f"{p_tok} {vs}"})
                            rows.append({"input": f"{canonical_map[tp]} {self.noise_gen.apply(best_vt)}", "target": f"{canonical_map[tp]} {best_vt}"})

            # ORFANI PURI
            for p, vals in s_lits.items():
                if p not in used_s:
                    for v in vals: rows.append({"input": f"{canonical_map[p]} {self.noise_gen.apply(v)}", "target": f"{canonical_map[p]} {v}"})
            for p, vals in t_lits.items():
                if p not in used_t:
                    for v in vals: rows.append({"input": f"{canonical_map[p]} {self.noise_gen.apply(v)}", "target": f"{canonical_map[p]} {v}"})

        return rows, canonical_map

    def _add_balanced_tasks(self, rows, p_tok, vs, vt):
        # noise(A) -> B, noise(B) -> A, A -> A, B -> B
        # Usiamo il noise generator robusto per stimolare parafrasi
        rows.append({"input": f"{p_tok} {self.noise_gen.apply(vs)}", "target": f"{p_tok} {vt}"})
        rows.append({"input": f"{p_tok} {self.noise_gen.apply(vt)}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vt}"})