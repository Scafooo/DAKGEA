import sys
import torch
import logging
import random
import time
import numpy as np
import re
import os
import string
from pathlib import Path
from collections import defaultdict, deque
from rdflib import Literal, URIRef
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
# USE CONTEXT INTERPOLATOR
from src.augmentation.methods.plm_context.mixup_context_interpolator import MixupContextInterpolator
# Re-use helpers
from src.augmentation.methods.plm.mixup_data_builder import clean_predicate, load_attr_names
from src.augmentation.methods.plm.scoring import calculate_pair_score

# --- CONFIGURAZIONE CONTEXT ---
MODEL_NAME = "google/flan-t5-large"
BATCH_SIZE = 8 # Reduced for context length
EPOCHS = 5 
TOTAL_REPORT_SAMPLES = 400 
MAX_ORPHANS_PER_PRED = 500
SWEEP_SAMPLES = 50
MAX_LEN_IN = 256 # Increased for context

torch.backends.cudnn.benchmark = True
logger = get_logger("FlanT5ContextSweep")

def get_context_str(graph, subject_uri, target_pred_uri, attr_map):
    """Extract context neighbors."""
    candidates = []
    for s, p, o in graph.triples((subject_uri, None, None)):
        if str(p) == str(target_pred_uri): continue
        
        p_name = clean_predicate(p, attr_map).replace('_', ' ')
        
        if isinstance(o, Literal):
            val = str(o).strip()
            if len(val) < 50: 
                candidates.append(f"<{p_name}>: {val}")
        elif isinstance(o, URIRef):
            if str(p).endswith("type"):
                local_type = str(o).split('/')[-1].split('#')[-1]
                candidates.append(f"<type>: {local_type}")
                
    if candidates:
        selected = random.sample(candidates, min(len(candidates), 3))
        return "; ".join(selected)
    return "generic"

def corrupt_text(text, noise_prob=0.15):
    """Introduce errors for denoising."""
    if len(text) < 3: return text
    chars = list(text)
    n_noise = max(1, int(len(chars) * noise_prob))
    for _ in range(n_noise):
        op = random.choice(['del', 'swap', 'sub'])
        idx = random.randint(0, len(chars) - 1)
        if op == 'del' and len(chars) > 2: del chars[idx]
        elif op == 'swap' and idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif op == 'sub':
            chars[idx] = random.choice(string.ascii_letters + string.digits)
    return "".join(chars)

def run_context_pipeline():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 CONTEXT-AWARE PIPELINE ".center(98) + "█"); print("█"*100)

    # 1. LOADING
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)
    
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target

    print("    [1/4] Building Context-Aware Training Set...")
    t5_rows = []
    
    # Pre-collect literals for fast access
    src_lits = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits[s].append((p, str(o)))
    tgt_lits = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits[s].append((p, str(o)))

    # Simple matching map (local name based)
    canonical_map = {}
    
    # A. ALIGNED PAIRS (Cross-KG translation with context)
    aligned_test_pool = defaultdict(list) # Stores (v1, v2, ctx1, ctx2) 
    
    for s_uri, t_uri in dataset.aligned_entities:
        s_attrs = src_lits.get(s_uri, [])
        t_attrs = tgt_lits.get(t_uri, [])
        
        for ps, vs in s_attrs:
            p_name = clean_predicate(ps, attr_map).replace('_', ' ')
            local_s = str(ps).split('/')[-1]
            
            for pt, vt in t_attrs:
                local_t = str(pt).split('/')[-1]
                
                # Loose matching on local name if canonical map empty
                if local_s == local_t:
                    # Context extraction
                    ctx_s = get_context_str(kg_src, s_uri, ps, attr_map)
                    ctx_t = get_context_str(kg_tgt, t_uri, pt, attr_map)
                    
                    # Formatted prompts
                    p1 = f"context: {ctx_s} | paraphrase the <{p_name}>: {vs}"
                    p2 = f"context: {ctx_t} | paraphrase the <{p_name}>: {vt}"
                    
                    t5_rows.append({"input": p1, "target": vt})
                    t5_rows.append({"input": p2, "target": vs})
                    
                    aligned_test_pool[p_name].append((vs, vt, ctx_s, ctx_t))
                    break

    # B. ORPHANS (Denoising with Context)
    orphans_by_pred = defaultdict(list)
    # Combine src and tgt iterators
    for kg, lits_dict in [(kg_src, src_lits), (kg_tgt, tgt_lits)]:
        for s_uri, attrs in lits_dict.items():
            for p, val in attrs:
                p_name = clean_predicate(p, attr_map).replace('_', ' ')
                # Context
                ctx = get_context_str(kg, s_uri, p, attr_map)
                orphans_by_pred[p_name].append((val, ctx))

    for p_name, items in orphans_by_pred.items():
        # Sample
        if len(items) > MAX_ORPHANS_PER_PRED:
            selected = random.sample(items, MAX_ORPHANS_PER_PRED)
        else:
            selected = items
            
        for val, ctx in selected:
            val_noisy = corrupt_text(val, noise_prob=0.15)
            prompt = f"context: {ctx} | paraphrase the <{p_name}>: {val_noisy}"
            t5_rows.append({"input": prompt, "target": val})

    random.shuffle(t5_rows)
    print(f"    Total training rows: {len(t5_rows)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_flan_t5_context_v1"
    
    # Initialize Context Interpolator
    interpolator = MixupContextInterpolator(
        model_name=MODEL_NAME, 
        out_dir=out_dir, 
        device=device,
        max_len_in=MAX_LEN_IN
    )

        # 2. TRAINING
        if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "adapter_model.safetensors").exists():
            print(f"    [2/4] Fine-tuning (Context-Aware, {len(t5_rows)} samples)...")
            # Use fine_tune from base class (it handles list of dicts fine)
            interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-5)
        else: 
            print(f"    [2/4] Context model ready.")
        # 3. SWEEP
    print(f"\n>>> PHASE 3: SWEEPING")
    sweep_pool = []
    all_test_preds = list(aligned_test_pool.keys())
    for _ in range(SWEEP_SAMPLES):
        if not all_test_preds: break
        p = random.choice(all_test_preds)
        if aligned_test_pool[p]:
            v1, v2, c1, c2 = random.choice(aligned_test_pool[p])
            sweep_pool.append((p, v1, v2, c1, c2))

    sweep_results = []
    for a in [0.3, 0.5]:
        for t in [0.7, 1.0]:
            interpolator.gen_temperature = t
            scs = []
            for p, v1, v2, c1, c2 in sweep_pool:
                # Use context in interpolation
                a1, a2 = interpolator.interpolate_pair(
                    v1, v2, predicate=p, alpha=a, 
                    context1=c1, context2=c2
                )
                res = calculate_pair_score(v1, v2, a1, a2)
                scs.append(res['score'])
            avg = np.mean(scs) if scs else 0
            sweep_results.append({"a": a, "t": t, "score": avg})
            print(f"      - Alpha={a} Temp={t} -> Score: {avg:.3f}")
    
    if sweep_results:
        best = sorted(sweep_results, key=lambda x: x['score'], reverse=True)[0]
    else:
        best = {"a": 0.5, "t": 0.7} # fallback
    print(f"    BEST CONFIG: {best}")

    # 4. REPORT
    print(f"    [4/4] Generating Context Report...")
    interpolator.gen_temperature = best['t']
    
    output_file = "massive_t5_context_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5 CONTEXT-AWARE REPORT\n")
        f.write(f"Config: {best} | Strategy: Context Injection\n")
        f.write("="*120 + "\n\n")
        
        f.write("SECTION 1: ALIGNED PAIRS (MIX-UP WITH CONTEXT)\n" + "-"*80 + "\n")
        gen_count = 0
        p_names = sorted(list(aligned_test_pool.keys()))
        while gen_count < TOTAL_REPORT_SAMPLES//2 and p_names:
            for p in p_names[:]:
                if not aligned_test_pool[p]: p_names.remove(p); continue
                v1, v2, c1, c2 = aligned_test_pool[p].pop(random.randrange(len(aligned_test_pool[p])))
                
                aa, ab = interpolator.interpolate_pair(
                    v1, v2, predicate=p, alpha=best['a'],
                    context1=c1, context2=c2
                )
                
                f.write(f"PAIR {gen_count+1:03d} | {p:20}\n")
                f.write(f"  CTX 1: {c1}\n")
                f.write(f"  V1:    {v1[:35]:35} -> AUG: {aa[:35]:35}\n")
                f.write(f"  CTX 2: {c2}\n")
                f.write(f"  V2:    {v2[:35]:35} -> AUG: {ab[:35]:35}\n")
                f.write("-" * 120 + "\n")
                gen_count += 1
                if gen_count >= TOTAL_REPORT_SAMPLES//2: break

        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES (REWRITE WITH CONTEXT)\n" + "-"*80 + "\n")
        o_count = 0
        o_p_names = sorted(list(orphans_by_pred.keys()))
        while o_count < TOTAL_REPORT_SAMPLES//2 and o_p_names:
            for p in o_p_names[:]:
                if not orphans_by_pred[p]: o_p_names.remove(p); continue
                val, ctx = orphans_by_pred[p].pop(random.randrange(len(orphans_by_pred[p])))
                
                aa, _ = interpolator.interpolate_pair(
                    val, val, predicate=p, alpha=0.5,
                    context1=ctx, context2=ctx
                )
                
                f.write(f"ORPHAN {o_count+1:03d} | {p:20}\n")
                f.write(f"  CTX:  {ctx}\n")
                f.write(f"  ORIG: {val[:45]:45} -> REWRITE: {aa[:45]:45}\n")
                f.write("-" * 80 + "\n")
                o_count += 1
                if o_count >= TOTAL_REPORT_SAMPLES//2: break
            
    print(f"\n>>> SUCCESS: Context Report saved to {output_file}")

if __name__ == "__main__":
    run_context_pipeline()
