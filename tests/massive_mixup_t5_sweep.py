import sys
import torch
import logging
import random
import time
import numpy as np
import re
from pathlib import Path
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

# --- CONFIGURAZIONE ---
MODEL_NAME = "t5-base"
SAMPLES_ALIGNED = 200 # Quante coppie mixup
SAMPLES_ORPHAN = 200  # Quanti orfani rewrite
torch.backends.cudnn.benchmark = True
logger = get_logger("T5FullReport")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_t5_full_report():
    print("\n" + "█"*100); print(f"█ RTX 4090: T5 REWRITE - FULL DIVERSE REPORT (ALIGNED + ORPHANS) ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=1000)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_t5_original_v1"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 2. SCANSIONE PER PREDICATI (ALIGNED & ORPHANS)
    print("    Organizing attributes by predicate for random sampling...")
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    aligned_entities = dataset.aligned_entities
    
    aligned_pairs_by_pred = defaultdict(list)
    orphans_by_pred = defaultdict(list)
    
    # Processo entità allineate
    for s_uri, t_uri in aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        
        # Coppie Allineate (Mix-up)
        matched_preds_s = set()
        for ps, vs in s_lits.items():
            p_can = canonical_map.get(ps, ps).split('/')[-1].split('#')[-1]
            for pt, vt in t_lits.items():
                if canonical_map.get(ps) == canonical_map.get(pt):
                    aligned_pairs_by_pred[p_can].append((vs, vt))
                    matched_preds_s.add(ps)
                    break
        
        # Orfani (Attributi in entità allineate che non hanno partner)
        for ps, vs in s_lits.items():
            if ps not in matched_preds_s:
                p_can = canonical_map.get(ps, ps).split('/')[-1].split('#')[-1]
                orphans_by_pred[p_can].append(vs)

    # 3. GENERAZIONE REPORT
    interpolator.latent_noise_std = 0.05
    interpolator.gen_temperature = 1.2
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA T5 FULL DIVERSE REPORT (ALIGNED + ORPHANS)\n")
        f.write("Selezione Random per garantire visibilità a Date, ID e predicati rari.\n")
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
            aligned_pairs_by_pred[p].remove((v1, v2))
            
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
            orphans_by_pred[p].remove(val)
            
            aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
            sim = util.cos_sim(semantic_model.encode(val), semantic_model.encode(aa)).item()
            
            f.write(f"ORPHAN {o_count+1:03d} | {p:20} | ORIG: {val[:40]:40} -> REWRITE: {aa[:40]:40} (Sim: {sim:.2f})\n")
            f.write("-" * 120 + "\n")
            o_count += 1
            if o_count % 10 == 0: f.flush()

    print(f"\n>>> SUCCESS: Full Diverse Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(time.time())
    run_t5_full_report()
