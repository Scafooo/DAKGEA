"""Training data builder for Mix-up BART with Coherent Value Pairing."""

import random
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from rdflib import Literal
from src.core.dataset import Dataset
from src.augmentation.methods.plm.services.attribute_matching_service import AttributeMatchingService

logger = logging.getLogger(__name__)

def _local_name(uri: str) -> str:
    return str(uri).split("/")[-1].split("#")[-1]

def _basic_noise(text: str) -> str:
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

    def build_training_data(self, dataset: Dataset, max_pairs_per_pred: int = 5000) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        logger.info("[MixupBuilder] Computing coherent correspondences...")
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
                            # ALLINEAMENTO SIMMETRICO (A <-> B)
                            # Insegniamo al modello che A e B sono la stessa cosa bidirezionalmente
                            rows.append({"input": f"{p_tok} {self.noise_gen.apply(vs)}", "target": f"{p_tok} {best_vt}"})
                            rows.append({"input": f"{p_tok} {self.noise_gen.apply(best_vt)}", "target": f"{p_tok} {vs}"})
                            
                            # IDENTITY (Ancoraggio)
                            rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
                            rows.append({"input": f"{p_tok} {best_vt}", "target": f"{p_tok} {best_vt}"})
                            
                            used_s.add(sp); used_t.add(tp)
                            pred_counts[p_tok] += 1

            # ORFANI PURI
            for p, vals in s_lits.items():
                if p not in used_s:
                    for v in vals: rows.append({"input": f"{canonical_map[p]} {_basic_noise(v)}", "target": f"{canonical_map[p]} {v}"})
            for p, vals in t_lits.items():
                if p not in used_t:
                    for v in vals: rows.append({"input": f"{canonical_map[p]} {_basic_noise(v)}", "target": f"{canonical_map[p]} {v}"})

        logger.info(f"[MixupBuilder] Built {len(rows)} training samples")
        return rows, canonical_map

    def _add_balanced_tasks(self, rows, p_tok, vs, vt):
        # 1-2. Identity
        rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vt}"})
        # 3-4. DAE
        rows.append({"input": f"{p_tok} {_basic_noise(vs)}", "target": f"{p_tok} {vs}"})
        rows.append({"input": f"{p_tok} {_basic_noise(vt)}", "target": f"{p_tok} {vt}"})
        # 5-6. Cross
        if vs.lower().strip() != vt.lower().strip():
            rows.append({"input": f"{p_tok} {vs}", "target": f"{p_tok} {vt}"})
            rows.append({"input": f"{p_tok} {vt}", "target": f"{p_tok} {vs}"})
