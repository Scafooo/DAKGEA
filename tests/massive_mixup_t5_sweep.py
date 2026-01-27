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

# --- CONFIGURAZIONE ---
MODEL_NAME = "t5-base"
TOTAL_SAMPLES = 400 
torch.backends.cudnn.benchmark = True
logger = get_logger("T5RandomOrphan")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_t5_random_report():
    print("\n" + "█"*100); print(f"█ RTX 4090: T5 REWRITE - RANDOM DIVERSE ORPHANS ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_t5_original_v1"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 2. SCANSIONE TUTTE LE TRIPLE (ORPHANS/GENERAL)
    print("    Collecting all predicates and values...")
    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target
    
    # Mappa: nome_predicato -> lista di valori unici
    predicate_map = defaultdict(set)
    
    all_triples = list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None)))
    for s, p, o in all_triples:
        if not isinstance(o, Literal): continue
        val = str(o).strip()
        if not val: continue
        p_name = str(p).split('/')[-1].split('#')[-1]
        predicate_map[p_name].add(val)

    # Convertiamo set in liste per il campionamento random
    all_preds = list(predicate_map.keys())
    for p in all_preds:
        predicate_map[p] = list(predicate_map[p])

    # 3. GENERAZIONE RANDOMIZZATA
    interpolator.latent_noise_std = 0.05
    interpolator.gen_temperature = 1.2
    
    output_file = "massive_t5_report.txt"
    print(f"    Total unique predicates available: {len(all_preds)}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA T5 RANDOM ORPHAN REPORT (REWRITE MODE)\n")
        f.write("Selezione Random: scelgo un predicato a caso, poi un suo valore a caso.\n")
        f.write("Garantisce che ID, Date e attributi rari appaiano quanto i nomi.\n")
        f.write("="*120 + "\n\n")
        
        gen_count = 0
        while gen_count < TOTAL_SAMPLES:
            # 1. Scegli un predicato a caso
            p_name = random.choice(all_preds)
            
            # 2. Scegli un valore a caso per quel predicato
            if not predicate_map[p_name]:
                all_preds.remove(p_name)
                if not all_preds: break
                continue
                
            val = random.choice(predicate_map[p_name])
            # Rimuoviamo il valore per non ripeterlo
            predicate_map[p_name].remove(val)
            
            # 3. Generazione Rewrite
            aa, _ = interpolator.interpolate_pair(val, val, predicate=p_name, alpha=0.5)
            
            # 4. Feedback e Scrittura
            sim = util.cos_sim(semantic_model.encode(val), semantic_model.encode(aa)).item()
            
            v_disp = (val[:50] + '..') if len(val) > 50 else val
            aa_disp = (aa[:50] + '..') if len(aa) > 50 else aa
            
            f.write(f"ORPHAN {gen_count+1:03d} | {p_name:20} | ORIG: {v_disp:52} -> REWRITE: {aa_disp:52} (Sim: {sim:.2f})\n")
            
            gen_count += 1
            if gen_count % 10 == 0: 
                f.flush()
                print(f"      Generated {gen_count}/{TOTAL_SAMPLES} samples...")

    print(f"\n>>> SUCCESS: Random Orphan Report saved to {output_file}")

if __name__ == "__main__":
    # Random puro basato sul tempo
    random.seed(time.time())
    run_t5_random_report()