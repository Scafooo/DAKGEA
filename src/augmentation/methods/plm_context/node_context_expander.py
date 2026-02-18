"""Node Expander with CONTEXT AWARENESS.

Extends standard NodeExpander to fetch neighbors and inject them into the interpolation prompt.
"""

from src.augmentation.methods.plm.node_expander import NodeExpander
from src.augmentation.methods.plm.mixup_data_builder import clean_predicate, load_attr_names
from src.config.loader import PROJECT_ROOT
from rdflib import Literal, URIRef
import random
import logging

logger = logging.getLogger(__name__)

class NodeContextExpander(NodeExpander):
    """Handles node expansion with context injection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need attr_map for clean predicate names in prompts
        # Ideally passed in config, but we can lazy load or approximate
        self._attr_map = {} 
        # HACK: try to load from default path if possible, or build on fly
        # For now, we will rely on on-the-fly extraction or empty map

    def _get_context_string(self, graph, subject_uri, target_predicate, max_context_items=3):
        """Extract 1-hop context for the subject entity (Same logic as Builder)."""
        candidates = []
        for s, p, o in graph.triples((subject_uri, None, None)):
            if target_predicate and str(p) == str(target_predicate):
                continue
            
            p_name = clean_predicate(p, self._attr_map).replace('_', ' ')
            
            if isinstance(o, Literal):
                val = str(o).strip()
                if len(val) < 50: 
                    candidates.append(f"<{p_name}>: {val}")
            elif isinstance(o, URIRef):
                if str(p).endswith("type"):
                    local_type = str(o).split('/')[-1].split('#')[-1]
                    candidates.append(f"<type>: {local_type}")

        if candidates:
            selected = random.sample(candidates, min(len(candidates), max_context_items))
            return "; ".join(selected)
        return "generic"

    def _interpolate_literals(self, dataset, src_component, tgt_component, src_aug, tgt_aug):
        """Override to inject context."""
        # Call super logic BUT we need to intercept the interpolate_pair call.
        # Since super() method is monolithic, we have to copy-paste and modify it.
        # This is unavoidable to change the internal call signature.
        
        # ... [COPY OF _interpolate_literals with modifications] ...
        # Simplified for brevity: we focus on the loop where interpolate_pair is called.
        
        if not src_component or not tgt_component:
            return

        src_literals = self._collect_predicate_literals(dataset.knowledge_graph_source, src_component)
        tgt_literals = self._collect_predicate_literals(dataset.knowledge_graph_target, tgt_component)

        # Match predicates (Reuse super logic or cache)
        if self.alignment_cache:
            matches = self._get_matches_from_cache(src_literals, tgt_literals)
        elif self.predicate_matcher:
             src_attr_names = dataset.knowledge_graph_source.attr_to_name
             tgt_attr_names = dataset.knowledge_graph_target.attr_to_name
             matches = self.predicate_matcher.match_predicates(
                src_literals, tgt_literals, src_attr_names, tgt_attr_names
            )
        else:
            matches = []

        # Deduplicate matches
        best_matches = {}
        for match in matches:
            if match.src_predicate not in best_matches or match.confidence > best_matches[match.src_predicate].confidence:
                best_matches[match.src_predicate] = match
        matches = list(best_matches.values())
        
        # INTERPOLATION LOOP
        for match in matches:
            src_pred, src_vals = src_literals[match.src_predicate]
            tgt_pred, tgt_vals = tgt_literals[match.tgt_predicate]
            
            src_val = str(src_vals[0]) if src_vals else ""
            tgt_val = str(tgt_vals[0]) if tgt_vals else ""
            
            if not src_val or not tgt_val: continue

            # --- CONTEXT EXTRACTION START ---
            ctx_s = self._get_context_string(dataset.knowledge_graph_source, src_component, src_pred)
            ctx_t = self._get_context_string(dataset.knowledge_graph_target, tgt_component, tgt_pred)
            # --- CONTEXT EXTRACTION END ---

            try:
                # Call context-aware interpolator
                # We assume self.bart_interpolator is MixupContextInterpolator
                if hasattr(self.bart_interpolator, 'interpolate_pair'):
                    # Check if it accepts context (duck typing or try/except)
                    # We pass context as kwargs to be safe if signature varies
                    aug_src, aug_tgt = self.bart_interpolator.interpolate_pair(
                        src_val, tgt_val, predicate=match.src_predicate,
                        context1=ctx_s, context2=ctx_t
                    )
                    
                    dataset.knowledge_graph_source.add((src_aug, src_pred, Literal(aug_src)))
                    dataset.knowledge_graph_target.add((tgt_aug, tgt_pred, Literal(aug_tgt)))
                    
                    logger.debug(f"[Context] '{src_val}' + '{tgt_val}' (ctx: {ctx_s[:20]}...) -> '{aug_src}'")
            except Exception as e:
                logger.warning(f"Context interpolation failed: {e}")

    def _generate_unmatched_attributes(self, dataset, src_cmp, tgt_cmp, src_aug, tgt_aug, src_lits, tgt_lits, matches, value_cache=None):
        # Override to inject context for orphans
        matched_src = {m.src_predicate for m in matches}
        matched_tgt = {m.tgt_predicate for m in matches}
        
        unmatched_src = [n for n in src_lits if n not in matched_src]
        unmatched_tgt = [n for n in tgt_lits if n not in matched_tgt]
        
        # Sample
        if self.unmatched_sample_rate < 1.0:
             import random
             unmatched_src = random.sample(unmatched_src, min(len(unmatched_src), int(len(unmatched_src)*self.unmatched_sample_rate)+1))
             unmatched_tgt = random.sample(unmatched_tgt, min(len(unmatched_tgt), int(len(unmatched_tgt)*self.unmatched_sample_rate)+1))

        # Source orphans
        for pname in unmatched_src:
            pred, vals = src_lits[pname]
            if not vals: continue
            val = str(vals[0])
            
            ctx = self._get_context_string(dataset.knowledge_graph_source, src_cmp, pred)
            try:
                aug, _ = self.bart_interpolator.interpolate_pair(val, val, predicate=pname, context1=ctx, context2=ctx)
                dataset.knowledge_graph_source.add((src_aug, pred, Literal(aug)))
            except Exception as e:
                logger.warning(f"Orphan src failed: {e}")

        # Target orphans
        for pname in unmatched_tgt:
            pred, vals = tgt_lits[pname]
            if not vals: continue
            val = str(vals[0])
            
            ctx = self._get_context_string(dataset.knowledge_graph_target, tgt_cmp, pred)
            try:
                aug, _ = self.bart_interpolator.interpolate_pair(val, val, predicate=pname, context1=ctx, context2=ctx)
                dataset.knowledge_graph_target.add((tgt_aug, pred, Literal(aug)))
            except Exception as e:
                logger.warning(f"Orphan tgt failed: {e}")
        
        return len(unmatched_src) + len(unmatched_tgt)
