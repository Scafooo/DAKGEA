"""Training data builder for Mix-up T5/BART with CONTEXT AWARENESS.

Injects structural context (neighbors) into the input prompt.
Prompt format:
Input: "context: <type>: Person; <birthDate>: 1980 | generate variation <name>: The Chemical Brothers"
Target: "Chemical Brothers"
"""

import random
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

from rdflib import Literal, URIRef
from src.core.dataset import Dataset

# Reuse helpers from original PLM module to keep code DRY
from src.augmentation.methods.plm.mixup_data_builder import (
    load_attr_names, clean_predicate, fix_unicode_escapes, 
    has_unicode_garbage, is_token_swap, is_only_filler_difference,
    min_edit_distance, extract_value_from_prompt, learn_variations_from_pairs,
    FLIP_PAIRS, FILLER_WORDS
)
from src.augmentation.methods.plm.creative_variation_generator import CreativeVariationGenerator

logger = logging.getLogger(__name__)

class MixupContextDataBuilder:
    """Build training data for Mix-up T5/BART with CONTEXT injection."""

    def __init__(self, confidence_threshold: float = 0.6, value_match_threshold: float = 0.3):
        self.value_match_threshold = value_match_threshold
        self.creative_gen = CreativeVariationGenerator()
        self._attr_map = {}

    def _get_context_string(self, graph, subject_uri, target_predicate, canonical_map, max_context_items=3):
        """Extract 1-hop context for the subject entity.
        
        Format: "<pred1>: val1; <pred2>: val2"
        Excludes the target predicate to avoid leakage/redundancy.
        """
        candidates = []
        
        # Query neighbors
        for s, p, o in graph.triples((subject_uri, None, None)):
            # Skip the predicate we are currently generating for
            if target_predicate and str(p) == str(target_predicate):
                continue
                
            # Clean predicate name for prompt
            p_name = clean_predicate(p, self._attr_map).replace('_', ' ')
            
            if isinstance(o, Literal):
                val = str(o).strip()
                if len(val) < 50: # Skip very long descriptions
                    candidates.append(f"<{p_name}>: {val}")
            
            elif isinstance(o, URIRef):
                # Handle rdf:type specifically
                if str(p).endswith("type"):
                    local_type = str(o).split('/')[-1].split('#')[-1]
                    candidates.append(f"<type>: {local_type}")
                else:
                    pass # Keep context simple (attributes + type)

        # Shuffle and limit
        if candidates:
            selected = random.sample(candidates, min(len(candidates), max_context_items))
            return "; ".join(selected)
        
        return "generic"

    def _build_canonical_map_from_matches(self, dataset: Dataset) -> Dict[str, str]:
        """Same as original."""
        canonical_map: Dict[str, str] = {}
        if dataset.attribute_matches:
            for src_uri, tgt_uris in dataset.attribute_matches.items():
                local = src_uri.split("/")[-1].split("#")[-1]
                token = f"<{local.upper()}>"
                canonical_map[src_uri] = token
                for tgt_uri in tgt_uris:
                    canonical_map[tgt_uri] = token
        
        all_predicates = (set(dataset.knowledge_graph_source.predicates()) |
                         set(dataset.knowledge_graph_target.predicates()))
        for p in all_predicates:
            p_str = str(p)
            if p_str not in canonical_map:
                from src.augmentation.methods.plm.mixup_data_builder import _local_name
                canonical_map[p_str] = f"<{_local_name(p)}>"
        return canonical_map

    def _filter_training_data(self, rows: List[Dict], min_edit_ratio: float = 0.1) -> List[Dict]:
        """Filter rows, handling context in input prompt."""
        filtered = []
        for row in rows:
            inp, tgt = row['input'], row['target']
            
            # Fix unicode
            tgt_fixed = fix_unicode_escapes(tgt)
            if tgt_fixed != tgt:
                tgt = tgt_fixed
                row['target'] = tgt_fixed

            # Parse value from prompt "context: ... | generate variation <pred>: VALUE"
            if "|" in inp:
                task_part = inp.split("|")[-1]
            else:
                task_part = inp
                
            inp_val = extract_value_from_prompt(task_part)

            if has_unicode_garbage(inp) or has_unicode_garbage(tgt): continue
            if is_token_swap(task_part, tgt): continue
            if is_only_filler_difference(task_part, tgt): continue

            max_len = max(len(inp_val), len(tgt))
            if max_len > 0:
                edit_dist = min_edit_distance(inp_val.lower(), tgt.lower())
                edit_ratio = edit_dist / max_len
                if edit_ratio < min_edit_ratio and inp_val.lower() != tgt.lower():
                    continue

            filtered.append(row)
        return filtered

    def build_training_data(self, dataset: Dataset, max_pairs_per_pred: int = 5000, dataset_path: str = None) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Build Context-Aware training data."""
        logger.info("[MixupContextBuilder] Building CONTEXT-AWARE training data...")

        if dataset_path:
            self._attr_map = load_attr_names(dataset_path)

        canonical_map = self._build_canonical_map_from_matches(dataset)
        
        kg_src = dataset.knowledge_graph_source
        kg_tgt = dataset.knowledge_graph_target

        # Collect literals map: URI -> [(pred, val), ...]
        src_lits_map = defaultdict(list)
        for s, p, o in kg_src.triples((None, None, None)):
            if isinstance(o, Literal):
                src_lits_map[s].append((p, str(o)))

        tgt_lits_map = defaultdict(list)
        for s, p, o in kg_tgt.triples((None, None, None)):
            if isinstance(o, Literal):
                tgt_lits_map[s].append((p, str(o)))

        rows = []
        pred_counts = defaultdict(int)
        
        def make_prompt(context, pred_name, value):
            return f"context: {context} | generate variation <{pred_name}>: {value}"

        # A. ALIGNED PAIRS
        for s_uri, t_uri in dataset.aligned_entities:
            s_attrs = src_lits_map.get(s_uri, [])
            t_attrs = tgt_lits_map.get(t_uri, [])
            
            # Fetch context (excluding current attributes dynamically below)
            # Optimization: Fetch base context once per entity? 
            # But context excludes *target* predicate, so it varies per example.
            
            for ps, vs in s_attrs:
                p_name = clean_predicate(ps, self._attr_map).replace('_', ' ')
                p_tok = canonical_map.get(str(ps), f"<{p_name}>")

                if pred_counts[p_tok] >= max_pairs_per_pred:
                    continue

                for pt, vt in t_attrs:
                    if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                        v1_c, v2_c = vs.strip().lower(), vt.strip().lower()

                        if v1_c != v2_c:
                            # Specific contexts excluding the current attribute
                            c_s = self._get_context_string(kg_src, s_uri, ps, canonical_map)
                            c_t = self._get_context_string(kg_tgt, t_uri, pt, canonical_map)
                            
                            # Real pairs x3
                            for _ in range(3):
                                rows.append({"input": make_prompt(c_s, p_name, vs), "target": vt})
                                rows.append({"input": make_prompt(c_t, p_name, vt), "target": vs})
                            
                            pred_counts[p_tok] += 1

                            # Synthetic
                            if random.random() < 0.3:
                                var_vs = self.creative_gen.generate(vs, vt, predicate=p_name)
                                if var_vs != vs and var_vs != vt:
                                    rows.append({"input": make_prompt(c_s, p_name, vs), "target": var_vs})

                            if random.random() < 0.3:
                                var_vt = self.creative_gen.generate(vt, vs, predicate=p_name)
                                if var_vt != vt and var_vt != vs:
                                    rows.append({"input": make_prompt(c_t, p_name, vt), "target": var_vt})
                        break

        # B. ORPHANS (with context)
        orphan_count = 0
        
        # Iterate entities to keep context
        for kg, lits_map in [(kg_src, src_lits_map), (kg_tgt, tgt_lits_map)]:
            for s_uri, attrs in lits_map.items():
                if not attrs: continue
                # Sample 1-2 attributes per entity
                targets = random.sample(attrs, min(len(attrs), 2))
                
                for p, val in targets:
                    p_name = clean_predicate(p, self._attr_map).replace('_', ' ')
                    
                    if random.random() < 0.1: # Downsample
                        v_creative = self.creative_gen.generate(val, predicate=p_name)
                        if v_creative != val and len(v_creative) > 2:
                            ctx = self._get_context_string(kg, s_uri, p, canonical_map)
                            rows.append({"input": make_prompt(ctx, p_name, val), "target": v_creative})
                            orphan_count += 1

        logger.info(f"[ORPHANS] Synthetic variations: {orphan_count}")

        # C. FLIP PAIRS (Generic context)
        for val, flipped in FLIP_PAIRS.items():
            for _ in range(3):
                rows.append({"input": f"context: generic | generate variation <value>: {val}", "target": flipped})

        # Filter
        rows = self._filter_training_data(rows)
        random.shuffle(rows)
        logger.info(f"Total Context-Aware samples: {len(rows)}")
        
        return rows, canonical_map
