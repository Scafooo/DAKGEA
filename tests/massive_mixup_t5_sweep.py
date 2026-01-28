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
from src.augmentation.methods.plm.scoring import calculate_score, calculate_score_detailed, calculate_pair_score

# --- CONFIGURAZIONE ---
MODEL_NAME = "google/flan-t5-large"
BATCH_SIZE = 16 
EPOCHS = 5 # Più epoche per forzare il distacco dall'identità
TOTAL_REPORT_SAMPLES = 400 
MAX_ORPHANS_PER_PRED = 300
SWEEP_SAMPLES = 50

torch.backends.cudnn.benchmark = True
logger = get_logger("FlanT5AntiBias")

def load_attr_names(dataset_path):
    attr_map = {}
    for i in [1, 2]:
        path = Path(dataset_path) / f"attribute_data/attr_names{i}"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2: attr_map[parts[0].strip()] = parts[1].strip()
    return attr_map

def clean_p(uri, attr_map):
    uri_str = str(uri)
    if uri_str in attr_map: return attr_map[uri_str].replace(' ', '_').lower()
    return uri_str.split('/')[-1].split('#')[-1].replace('>', '').replace('<', '').lower()

# Scoring importato da src.augmentation.methods.plm.scoring

def run_antibias_t5_pipeline():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 LARGE - ANTI-BIAS TRAINING ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)
    
    builder = MixupDataBuilder()
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=10)
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target

    print("    [1/4] Building Anti-Bias training set (Changes Only)...")
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
            p_name = clean_p(ps, attr_map).replace('_', ' ')
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    # AGGIUNGIAMO SOLO SE DIVERSI (Anti-Bias)
                    if vs.lower().strip() != vt.lower().strip():
                        t5_rows.append({"input": f"{p_name} | {vs}", "target": vt})
                        t5_rows.append({"input": f"{p_name} | {vt}", "target": vs})
                    aligned_test_pool[p_name].append((vs, vt))
                    break
    
    orphans_by_pred = defaultdict(list)
    for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
        if isinstance(o, Literal):
            val = str(o).strip()
            if val:
                p_name = clean_p(p, attr_map).replace('_', ' ')
                orphans_by_pred[p_name].append(val)
    
    for p_name, vals in orphans_by_pred.items():
        unique_vals = list(set(vals))
        selected = random.sample(unique_vals, min(len(unique_vals), MAX_ORPHANS_PER_PRED))
        for v in selected:
            # Per gli orfani, alleniamo con identità ma anche con un po' di noise per generalizzare
            t5_rows.append({"input": f"{p_name} | {v}", "target": v})

    random.shuffle(t5_rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_flan_t5_large_antibias_v1"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 2. TRAINING
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    [2/4] Fine-tuning (Anti-Bias, {len(t5_rows)} samples)...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-5, force_retrain=True)
    else: print(f"    [2/4] Anti-bias model ready.")

    # 3. SWEEP
    print(f"\n>>> PHASE 3: SWEEPING FOR MAXIMUM DIVERSITY")
    sweep_pool = []
    all_test_preds = list(aligned_test_pool.keys())
    for _ in range(SWEEP_SAMPLES):
        p = random.choice(all_test_preds)
        v1, v2 = random.choice(aligned_test_pool[p])
        sweep_pool.append((p, v1, v2))

    sweep_results = []
    for a in [0.3, 0.5]:
        for n in [0.05, 0.1]: # Alziamo il rumore per stanare l'identità
            for t in [0.8, 1.2]: # Alziamo la temperatura
                interpolator.latent_noise_std, interpolator.gen_temperature = n, t
                scs = []
                for p, v1, v2 in sweep_pool:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                    # Score con bonus coerenza cross-output
                    pair_result = calculate_pair_score(v1, v2, a1, a2)
                    scs.append(pair_result["score"])
                avg = np.mean(scs)
                sweep_results.append({"a": a, "n": n, "t": t, "score": avg})
                print(f"      - Alpha={a} Noise={n} Temp={t} -> Score: {avg:.3f}")
    
    best = sorted(sweep_results, key=lambda x: x['score'], reverse=True)[0]
    print(f"    BEST CONFIG: {best}")

    # 4. FINAL REPORT
    print(f"    [4/4] Generating anti-bias report...")
    interpolator.latent_noise_std = best['n']
    interpolator.gen_temperature = best['t']
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5 ANTI-BIAS REPORT\n")
        f.write(f"Config: {best} | Model: Large | Strategy: Filtered Identical Pairs\n")
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

        # SECTION 2: ORPHAN ATTRIBUTES (DIVERSE REWRITE)
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES (DIVERSE REWRITE)\n" + "-"*80 + "\n")
        o_count = 0
        o_p_names = sorted(list(orphans_by_pred.keys()))
        while o_count < TOTAL_REPORT_SAMPLES//2 and o_p_names:
            for p in o_p_names[:]:
                if not orphans_by_pred[p]: o_p_names.remove(p); continue
                val = orphans_by_pred[p].pop(random.randrange(len(orphans_by_pred[p])))
                aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
                
                # Usa il nuovo scoring centralizzato anche qui
                score = calculate_score(val, aa)
                
                f.write(f"ORPHAN {o_count+1:03d} | {p:20} | ORIG: {val[:45]:45} -> REWRITE: {aa[:45]:45} (Score: {score:.2f})\n")
                o_count += 1
                if o_count >= TOTAL_REPORT_SAMPLES//2: break
            
    print(f"\n>>> SUCCESS: Anti-bias Report saved to {output_file}")

if __name__ == "__main__":
    run_antibias_t5_pipeline()