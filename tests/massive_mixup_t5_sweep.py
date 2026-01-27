import sys
import torch
import logging
import random
import time
import numpy as np
import re
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

# --- CONFIGURAZIONE ---
MODEL_NAME = "google/flan-t5-base"
BATCH_SIZE = 32
EPOCHS = 10
SAMPLES_ALIGNED = 200 
SAMPLES_ORPHAN = 200  
torch.backends.cudnn.benchmark = True
logger = get_logger("FlanT5Diverse")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_t5_full_report():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 REWRITE - TRAINING + DIVERSE REPORT ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    print("    [1/4] Loading BBC_DB and building training pairs...")
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=50000)
    
    # Preparazione righe per Flan-T5
    t5_rows = []
    for r in train_rows:
        p_tok = r['input'].split(' ')[0]
        p_name = p_tok.replace("<", "").replace(">", "").lower()
        inp_val = r['input'].replace(p_tok, "").strip()
        tgt_val = r['target'].replace(p_tok, "").strip()
        t5_rows.append({"input": f"rewrite {p_name}: {inp_val}", "target": tgt_val})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_flan_t5_v1"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 2. TRAINING (Se necessario)
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    [2/4] Starting Flan-T5 Fine-tuning (REWRITE prompt, {len(t5_rows)}} rows)...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, force_retrain=True)
    else:
        print(f"    [2/4] Found existing Flan-T5 model, skipping training.")

    # 3. EXTRAZIONE ATTRIBUTI (Ottimizzata con pre-indexing)
    print("    [3/4] Organizing attributes for report (Pre-indexing literals)...")
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    
    # Pre-indicizziamo i letterali per velocità
    src_lits_map = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits_map[s].append((str(p), str(o)))
    
    tgt_lits_map = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits_map[s].append((str(p), str(o)))

    aligned_pairs_by_pred = defaultdict(list)
    orphans_by_pred = defaultdict(list)
    
    for s_uri, t_uri in dataset.aligned_entities:
        s_lits = src_lits_map.get(s_uri, [])
        t_lits = tgt_lits_map.get(t_uri, [])
        
        matched_preds_s = set()
        for ps, vs in s_lits:
            p_can = canonical_map.get(ps, ps).split('/')[-1].split('#')[-1]
            for pt, vt in t_lits:
                if canonical_map.get(ps) == canonical_map.get(pt):
                    aligned_pairs_by_pred[p_can].append((vs, vt))
                    matched_preds_s.add(ps)
                    break
        
        # Orfani
        for ps, vs in s_lits:
            if ps not in matched_preds_s:
                p_can = canonical_map.get(ps, ps).split('/')[-1].split('#')[-1]
                orphans_by_pred[p_can].append(vs)

    # 4. GENERAZIONE REPORT
    print(f"    [4/4] Generating balanced report (randomized predicates)...")
    interpolator.latent_noise_std = 0.05
    interpolator.gen_temperature = 1.2
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5 FULL DIVERSE REPORT (ALIGNED + ORPHANS)\n")
        f.write("Selection: Random Predicates (Ensures Date/ID visibility)\n")
        f.write("="*120 + "\n\n")
        
        # --- SEZIONE 1: ALIGNED ---
        f.write("SECTION 1: ALIGNED PAIRS (MIX-UP INTERPOLATION)\n" + "-"*80 + "\n")
        gen_count = 0
        all_p_aligned = list(aligned_pairs_by_pred.keys())
        while gen_count < SAMPLES_ALIGNED and all_p_aligned:
            p = random.choice(all_p_aligned)
            if not aligned_pairs_by_pred[p]:
                all_p_aligned.remove(p); continue
            v1, v2 = random.choice(aligned_pairs_by_pred[p])
            aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
            f.write(f"PAIR {gen_count+1:03d} | {p:20} | V1: {v1[:35]:35} -> AUG: {aa[:35]:35}\n")
            f.write(f"         | {' ':20} | V2: {v2[:35]:35} -> AUG: {ab[:35]:35}\n")
            f.write("-" * 120 + "\n")
            gen_count += 1
            if gen_count % 10 == 0: f.flush()

        # --- SEZIONE 2: ORPHANS ---
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES (REWRITE MODE)\n" + "-"*80 + "\n")
        o_count = 0
        all_p_orphan = list(orphans_by_pred.keys())
        while o_count < SAMPLES_ORPHAN and all_p_orphan:
            p = random.choice(all_p_orphan)
            if not orphans_by_pred[p]:
                all_p_orphan.remove(p); continue
            val = random.choice(orphans_by_pred[p])
            aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
            sim = util.cos_sim(semantic_model.encode(val), semantic_model.encode(aa)).item()
            f.write(f"ORPHAN {o_count+1:03d} | {p:20} | ORIG: {val[:40]:40} -> REWRITE: {aa[:40]:40} (Sim: {sim:.2f})\n")
            f.write("-" * 120 + "\n")
            o_count += 1
            if o_count % 10 == 0: f.flush()

    print(f"\n>>> SUCCESS: Flan-T5 Diverse Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(time.time())
    run_t5_full_report()