import sys
import torch
import logging
import random
import time
import numpy as np
import re
import os
from pathlib import Path
from collections import defaultdict, deque
from rdflib import Literal
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_t5_interpolator import MixupT5Interpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE DEFINITIVA ---
MODEL_NAME = "google/flan-t5-large"
BATCH_SIZE = 16 
EPOCHS = 5 # Più epoche per stabilità su Large
TOTAL_REPORT_SAMPLES = 400 
MAX_ORPHANS_PER_PRED = 500
SWEEP_SAMPLES = 50

torch.backends.cudnn.benchmark = True
logger = get_logger("FlanT5Final")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_score(orig, gen):
    gen_l, orig_l = gen.lower().strip(), orig.lower().strip()
    if len(gen_l) < 1: return -1.0
    if any(x in gen_l for x in ["rewrite", "paraphrase", ":"]): return -1.0
    
    emb_orig = semantic_model.encode(orig_l, convert_to_tensor=True)
    emb_gen = semantic_model.encode(gen_l, convert_to_tensor=True)
    sim = util.cos_sim(emb_orig, emb_gen).item()
    
    if sim < 0.75: return -1.0 # Più severo sulla qualità semantica
    if gen_l == orig_l: return 0.5 # Penalità per identità
    return sim * 2.0

def run_final_t5_pipeline():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 LARGE - FINAL STABLE PIPELINE ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    
    builder = MixupDataBuilder()
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=10)
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target

    print("    [1/4] Preparing training data (Balanced Aligned + Orphans)...")
    t5_rows = []
    src_lits = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits[s].append((p, str(o)))
    tgt_lits = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits[s].append((p, str(o)))

    aligned_test_pool = defaultdict(list)
    for s_uri, t_uri in dataset.aligned_entities:
        s_attrs = src_lits.get(s_uri, [])
        t_attrs = tgt_lits.get(t_uri, [])
        for ps, vs in s_attrs:
            p_name = str(ps).split('/')[-1].split('#')[-1].lower()
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    # Formato Prompt Semplificato: "rewrite: [v]"
                    t5_rows.append({"input": f"rewrite: {vs}", "target": vt})
                    t5_rows.append({"input": f"rewrite: {vt}", "target": vs})
                    aligned_test_pool[p_name].append((vs, vt))
                    break
    
    orphans_by_pred = defaultdict(list)
    for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
        if isinstance(o, Literal):
            val = str(o).strip()
            if val:
                p_name = str(p).split('/')[-1].split('#')[-1].lower()
                orphans_by_pred[p_name].append(val)
    
    for p_name, vals in orphans_by_pred.items():
        selected = random.sample(list(set(vals)), min(len(set(vals)), MAX_ORPHANS_PER_PRED))
        for v in selected:
            t5_rows.append({"input": f"rewrite: {v}", "target": v})

    random.shuffle(t5_rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_flan_t5_large_final"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 2. TRAINING
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    [2/4] Fine-tuning (5 Epochs, LR=3e-5)...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-5, force_retrain=True)
    else: print(f"    [2/4] Model ready.")

    # 3. SWEEP (Ricerca stabilità)
    print(f"\n>>> PHASE 3: SWEEPING FOR STABILITY")
    sweep_pool = []
    all_test_preds = list(aligned_test_pool.keys())
    for _ in range(SWEEP_SAMPLES):
        p = random.choice(all_test_preds)
        v1, v2 = random.choice(aligned_test_pool[p])
        sweep_pool.append((p, v1, v2))

    sweep_results = []
    for a in [0.3, 0.5]:
        for n in [0.0, 0.02]:
            for t in [0.7, 1.0]:
                interpolator.latent_noise_std, interpolator.gen_temperature = n, t
                scs = []
                for p, v1, v2 in sweep_pool:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                    scs.append((calculate_score(v1, a1) + calculate_score(v2, a2))/2)
                avg = np.mean(scs)
                sweep_results.append({"a": a, "n": n, "t": t, "score": avg})
                print(f"      - Alpha={a} Noise={n} Temp={t} -> Score: {avg:.3f}")
    
    best = sorted(sweep_results, key=lambda x: x['score'], reverse=True)[0]
    print(f"    BEST CONFIG: {best}")

    # 4. FINAL REPORT
    print(f"    [4/4] Generating high-quality report...")
    interpolator.latent_noise_std = best['n']
    interpolator.gen_temperature = best['t']
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5 FINAL STABLE REPORT\n")
        f.write(f"Config: {best} | Model: Large | Masking: Combined\n")
        f.write("="*120 + "\n\n")
        
        f.write("SECTION 1: ALIGNED PAIRS (DIVERSE MIX-UP)\n" + "-"*80 + "\n")
        gen_count = 0
        p_names = sorted(list(aligned_test_pool.keys()))
        while gen_count < TOTAL_REPORT_SAMPLES//2 and p_names:
            for p in p_names[:]:
                if not aligned_test_pool[p]: p_names.remove(p); continue
                v1, v2 = aligned_test_pool[p].pop(random.randrange(len(aligned_test_pool[p])))
                aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best['a'])
                f.write(f"PAIR {gen_count+1:03d} | {p:20} | V1: {v1[:35]:35} -> AUG: {aa[:35]:35}\n")
                f.write(f"         | {' ':20} | V2: {v2[:35]:35} -> AUG: {ab[:35]:35}\n")
                f.write("-" * 120 + "\n")
                gen_count += 1
                if gen_count >= TOTAL_REPORT_SAMPLES//2: break

        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES (DIVERSE REWRITE)\n" + "-"*80 + "\n")
        o_count = 0
        o_p_names = sorted(list(orphans_by_pred.keys()))
        while o_count < TOTAL_REPORT_SAMPLES//2 and o_p_names:
            for p in o_p_names[:]:
                if not orphans_by_pred[p]: o_p_names.remove(p); continue
                val = orphans_by_pred[p].pop(random.randrange(len(orphans_by_pred[p])))
                aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
                sim = util.cos_sim(semantic_model.encode(val), semantic_model.encode(aa)).item()
                f.write(f"ORPHAN {o_count+1:03d} | {p:20} | ORIG: {val[:45]:45} -> REWRITE: {aa[:45]:45} (Sim: {sim:.2f})\n")
                o_count += 1
                if o_count >= TOTAL_REPORT_SAMPLES//2: break
            
    print(f"\n>>> SUCCESS: Report saved to {output_file}")

if __name__ == "__main__":
    run_final_t5_pipeline()
