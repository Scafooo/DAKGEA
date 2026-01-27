import sys
import torch
import logging
import random
import time
import numpy as np
import re
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict
from rdflib import Literal
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_t5_interpolator import MixupT5Interpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE T5 ---
MODEL_NAME = "t5-base"
BATCH_SIZE = 32
EPOCHS = 10 
SAMPLES_ALIGNED = 200 # Totale coppie mixup nel report
SAMPLES_ORPHAN = 200  # Totale orfani nel report (stratificati)
SWEEP_SAMPLES = 50

torch.backends.cudnn.benchmark = True
logger = get_logger("T5Sweep")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_score(orig, gen):
    gen_l, orig_l = gen.lower().strip(), orig.lower().strip()
    if len(gen_l) < 1: return -1.0
    emb_orig = semantic_model.encode(orig_l, convert_to_tensor=True)
    emb_gen = semantic_model.encode(gen_l, convert_to_tensor=True)
    sim = util.cos_sim(emb_orig, emb_gen).item()
    if sim < 0.4: return -1.0 
    score = sim * 2.0
    if set(gen_l.split()) - set(orig_l.split()): score += 1.5
    return score

def run_t5_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: T5 REWRITE STRATIFIED (DATES, IDS, NAMES) ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=50000)
    
    t5_rows = []
    for r in train_rows:
        p_tok = r['input'].split(' ')[0]
        p_name = p_tok.replace("<", "").replace(">", "").lower()
        inp_val = r['input'].replace(p_tok, "").strip()
        tgt_val = r['target'].replace(p_tok, "").strip()
        t5_rows.append({"input": f"rewrite {p_name}: {inp_val}", "target": tgt_val})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_t5_original_v1"

    # 2. TRAINING
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting T5 Training with REWRITE prompt...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, force_retrain=True)
    else:
        print(f"    [RESUME] Found existing T5 model.")

    # 3. EXTRAZIONE STRATIFICATA (Per includere ID, Date, ecc.)
    print("    Extracting stratified attributes...")
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    aligned_entities = dataset.aligned_entities
    
    test_by_pred = defaultdict(list)
    orphan_by_pred = defaultdict(list)
    
    # Estrazione da entità allineate
    for s_uri, t_uri in aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        
        for ps, vs in s_lits.items():
            p_can = canonical_map.get(ps, ps).replace("<", "").replace(">", "")
            matched = False
            for pt, vt in t_lits.items():
                if canonical_map.get(ps) == canonical_map.get(pt):
                    test_by_pred[p_can].append((vs, vt))
                    matched = True
                    break
            if not matched:
                orphan_by_pred[p_can].append(vs)

    # 4. PARAMETER SWEEP
    print(f"\n>>> PHASE 2: PARAMETER SWEEP (on mixed predicates)")
    sweep_pool = []
    all_p = list(test_by_pred.keys())
    for _ in range(SWEEP_SAMPLES):
        p = random.choice(all_p)
        v1, v2 = random.choice(test_by_pred[p])
        sweep_pool.append((p, v1, v2))

    results = []
    for a in [0.3, 0.5]:
        for n in [0.0, 0.05]:
            for t in [1.0, 1.3]:
                interpolator.latent_noise_std, interpolator.gen_temperature = n, t
                scs = []
                for p, v1, v2 in sweep_pool:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                    scs.append((calculate_score(v1, a1) + calculate_score(v2, a2))/2)
                avg = np.mean(scs)
                results.append({"a": a, "n": n, "t": t, "score": avg})
                print(f"      - Alpha={a} Noise={n} Temp={t} -> Score: {avg:.3f}")
    
    best = sorted(results, key=lambda x: x['score'], reverse=True)[0]

    # 5. GENERATION & REPORT STRATIFICATO
    print("\n" + "="*100); print(" GENERATING DIVERSE REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_temperature = best['t']
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA T5 STRATIFIED REPORT | Config: {best}\n")
        f.write(f"MODEL: {MODEL_NAME} | PROMPT: rewrite\n")
        f.write("="*120 + "\n\n")
        
        # SECTION 1: ALIGNED
        f.write("SECTION 1: ALIGNED PAIRS (STRATIFIED)\n" + "-"*80 + "\n")
        count = 0
        p_list = sorted(list(test_by_pred.keys()))
        while count < SAMPLES_ALIGNED and p_list:
            for p in p_list[:]: # Copia per rimozione
                if not test_by_pred[p]: 
                    p_list.remove(p); continue
                v1, v2 = test_by_pred[p].pop(random.randrange(len(test_by_pred[p])))
                aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best['a'])
                f.write(f"PAIR {count+1:03d} | {p[:20]:20} | V1: {v1[:35]:35} -> AUG: {aa[:35]:35}\n")
                f.write(f"         | {' ':20} | V2: {v2[:35]:35} -> AUG: {ab[:35]:35} | Voto:[ ]/5\n")
                f.write("-" * 120 + "\n")
                count += 1
                if count >= SAMPLES_ALIGNED: break
            
        # SECTION 2: ORPHANS (DATE, ID, etc.)
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES (STRATIFIED - DATES, IDS, ETC)\n" + "-"*80 + "\n")
        o_count = 0
        o_p_list = sorted(list(orphan_by_pred.keys()))
        while o_count < SAMPLES_ORPHAN and o_p_list:
            for p in o_p_list[:]:
                if not orphan_by_pred[p]:
                    o_p_list.remove(p); continue
                v = orphan_by_pred[p].pop(random.randrange(len(orphan_by_pred[p])))
                # Rewrite orfano (alpha 0.5 su se stesso)
                aa, _ = interpolator.interpolate_pair(v, v, predicate=p, alpha=0.5)
                sim = util.cos_sim(semantic_model.encode(v), semantic_model.encode(aa)).item()
                f.write(f"ORPHAN {o_count+1:03d} | {p[:20]:20} | ORIG: {v[:40]:40} -> REWRITE: {aa[:40]:40} (Sim: {sim:.2f}) | Voto:[ ]/5\n")
                f.write("-" * 120 + "\n")
                o_count += 1
                if o_count >= SAMPLES_ORPHAN: break

    print(f">>> SUCCESS: Diverse T5 Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_t5_sweep()
