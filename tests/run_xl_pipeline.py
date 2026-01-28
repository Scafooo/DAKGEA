import sys
import torch
import logging
import random
import time
import numpy as np
import re
import os
from pathlib import Path
from collections import defaultdict
from rdflib import Literal
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder
from src.augmentation.methods.plm.scoring import calculate_pair_score, calculate_score
from src.augmentation.methods.plm.noise_generator import NoiseGenerator

# --- CONFIGURAZIONE STABILE XL BF16 ---
MODEL_NAME = "google/flan-t5-xl"
BATCH_SIZE = 8 
EPOCHS = 3
TOTAL_REPORT_SAMPLES = 200 
MAX_ORPHANS_PER_PRED = 200
SWEEP_SAMPLES = 50

logger = get_logger("FlanT5XL_Stable")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
noise_gen = NoiseGenerator()

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

def run_xl_pipeline():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 XL (3B) - STABLE DENOISING PIPELINE ".center(98) + "█"); print("█"*100)

    # 1. DATA
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)
    builder = MixupDataBuilder()
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=10)
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target

    print("    [1/4] Building Training Data (Denoising Focus)...")
    t5_rows = []
    src_lits = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits[s].append((p, str(o)))
    tgt_lits = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits[s].append((p, str(o)))

    # A. Aligned
    aligned_test_pool = defaultdict(list)
    for s_uri, t_uri in dataset.aligned_entities:
        s_attrs = src_lits.get(s_uri, [])
        t_attrs = tgt_lits.get(t_uri, [])
        for ps, vs in s_attrs:
            p_name = clean_p(ps, attr_map).replace('_', ' ')
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    v1_c, v2_c = vs.strip().lower(), vt.strip().lower()
                    if v1_c != v2_c:
                        # TRAINING: Impara trasformazioni reali cross-KG
                        t5_rows.append({"input": f"generate synthetic variation <{p_name}>: {vs}", "target": vt})
                        t5_rows.append({"input": f"generate synthetic variation <{p_name}>: {vt}", "target": vs})
                    aligned_test_pool[p_name].append((vs, vt))
                    break
    
    # B. Orphans (Denoising Puro: Sporco -> Pulito)
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
            # USA LA LOGICA STABILE: Corrompiamo l'input, il target è l'originale
            v_noisy = noise_gen.apply(v, predicate=p_name)
            if v_noisy != v:
                t5_rows.append({"input": f"generate synthetic variation <{p_name}>: {v_noisy}", "target": v})

    random.shuffle(t5_rows)
    print(f"    Total training samples: {len(t5_rows)}")

    # 2. MODEL XL (BF16 + LoRA)
    device = "cuda"
    out_dir = "./results/t5_xl_bf16_stable_v1"
    interpolator = MixupT5XLInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 3. TRAINING
    if not (Path(out_dir) / "adapter_model.bin").exists() and not (Path(out_dir) / "adapter_model.safetensors").exists():
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3)
    else: print("    [2/4] XL Adapters found.")

    # 3. SWEEP (Range ridotti e stabili)
    print(f"\n>>> PHASE 3: SWEEPING")
    sweep_pool = []
    all_test_preds = list(aligned_test_pool.keys())
    for _ in range(min(SWEEP_SAMPLES, len(all_test_preds))):
        p = random.choice(all_test_preds)
        v1, v2 = random.choice(aligned_test_pool[p])
        sweep_pool.append((p, v1, v2))

    sweep_results = []
    for a in [0.3, 0.5]:
        for n in [0.0, 0.02]: # Rumore latente molto basso per evitare garbage
            for t in [0.7, 1.0]:
                interpolator.latent_noise_std, interpolator.gen_temperature = n, t
                scs = []
                for p, v1, v2 in sweep_pool:
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                    res = calculate_pair_score(v1, v2, aa, ab)
                    scs.append(res['score'])
                avg = np.mean(scs)
                sweep_results.append({"a": a, "n": n, "t": t, "score": avg})
                print(f"      - Alpha={a} Noise={n} Temp={t} -> Score: {avg:.3f}")
    
    best = sorted(sweep_results, key=lambda x: x['score'], reverse=True)[0]
    print(f"    BEST CONFIG: {best}")

    # 4. REPORT
    print(f"    [4/4] Generating Stable XL Report...")
    interpolator.latent_noise_std, interpolator.gen_temperature = best['n'], best['t']
    output_file = "massive_t5_xl_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5-XL (3B) STABLE BF16 REPORT\n")
        f.write(f"Config: {best} | Model: XL | Strategy: Denoising Structural Training\n")
        f.write("="*120 + "\n\n")
        
        # Aligned
        f.write("SECTION 1: ALIGNED MIXUP (XL)\n" + "-"*80 + "\n")
        p_names = sorted(list(aligned_test_pool.keys()))
        count = 0
        while count < TOTAL_REPORT_SAMPLES // 2 and p_names:
            for p in p_names[:]:
                if not aligned_test_pool[p]: p_names.remove(p); continue
                v1, v2 = aligned_test_pool[p].pop(random.randrange(len(aligned_test_pool[p])))
                aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best['a'])
                res = calculate_pair_score(v1, v2, aa, ab)
                f.write(f"PAIR {count+1:03d} | {p:20} | V1: {v1[:35]:35} -> AUG: {aa[:35]:35}\n")
                f.write(f"         | Score: {res['score']:.2f}        | V2: {v2[:35]:35} -> AUG: {ab[:35]:35} | Voto:[ ]/5\n")
                f.write("-" * 120 + "\n")
                count += 1
                if count >= TOTAL_REPORT_SAMPLES // 2: break

        # Orphans
        f.write("\nSECTION 2: ORPHAN VARIATIONS (XL)\n" + "-"*80 + "\n")
        o_names = sorted(list(orphans_by_pred.keys()))
        o_count = 0
        while o_count < TOTAL_REPORT_SAMPLES // 2 and o_names:
            for p in o_names[:]:
                if not orphans_by_pred[p]: o_names.remove(p); continue
                val = orphans_by_pred[p].pop(random.randrange(len(orphans_by_pred[p])))
                # Per generare una variazione, usiamo il modello allenato al denoising
                # ma con temperatura > 0, che lo forza a non essere puramente conservativo.
                aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
                score = calculate_score(val, aa)
                f.write(f"ORPHAN {o_count+1:03d} | {p:20} | ORIG: {val[:45]:45} -> VAR: {aa[:45]:45} (Score: {score:.2f}) | Voto:[ ]/5\n")
                o_count += 1
                if o_count >= TOTAL_REPORT_SAMPLES // 2: break

    print(f"\n>>> SUCCESS: XL Stable Report saved to {output_file}")

if __name__ == "__main__":
    run_xl_pipeline()
