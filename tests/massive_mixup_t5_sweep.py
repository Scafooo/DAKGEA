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
torch.backends.cudnn.benchmark = True
logger = get_logger("T5Exhaustive")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_t5_exhaustive_report():
    print("\n" + "█"*100); print(f"█ RTX 4090: T5 REWRITE - EXHAUSTIVE PREDICATE REPORT ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_t5_original_v1"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 2. SCANSIONE TOTALE DEI PREDICATI
    print("    Scanning entire KG for ALL predicates...")
    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target
    
    # Raggruppiamo TUTTI i valori per ogni predicato
    predicate_data = defaultdict(list)
    
    for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
        if not isinstance(o, Literal): continue
        val = str(o).strip()
        if not val: continue
        
        # Prendiamo il nome corto del predicato
        p_uri = str(p)
        p_name = p_uri.split('/')[-1].split('#')[-1]
        predicate_data[p_name].append(val)

    # 3. GENERAZIONE REPORT ESAUSTIVO
    # Usiamo parametri di generazione bilanciati
    interpolator.latent_noise_std = 0.05
    interpolator.gen_temperature = 1.2
    interpolator.gen_num_beams = 4
    
    output_file = "massive_t5_report.txt"
    print(f"    Found {len(predicate_data)} unique predicates. Generating samples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA T5 EXHAUSTIVE ATTRIBUTE REPORT (REWRITE MODE)\n")
        f.write("Questo report contiene esempi per OGNI predicato trovato nel dataset.\n")
        f.write("="*120 + "\n\n")
        
        # Ordiniamo i predicati alfabeticamente per facilitare la consultazione
        sorted_preds = sorted(predicate_data.keys())
        
        total_generated = 0
        for p_name in sorted_preds:
            vals = list(set(predicate_data[p_name])) # Rimuoviamo duplicati esatti
            random.shuffle(vals)
            
            f.write(f"\nPREDICATE: {p_name} ({len(vals)} unique values found)\n")
            f.write("-" * 60 + "\n")
            
            # Mostriamo fino a 15 esempi per ogni predicato (per non rendere il file infinito ma essere esaustivi)
            num_samples = min(15, len(vals))
            for i in range(num_samples):
                v = vals[i]
                # Generazione Rewrite
                # Nota: usiamo alpha=0.5 su se stesso per ottenere una parafrasi/rewrite del valore singolo
                aa, _ = interpolator.interpolate_pair(v, v, predicate=p_name, alpha=0.5)
                
                # Calcolo sim per feedback
                emb_v = semantic_model.encode(v)
                emb_aa = semantic_model.encode(aa)
                sim = util.cos_sim(emb_v, emb_aa).item()
                
                # Troncamento per visualizzazione pulita nel report
                v_disp = (v[:60] + '..') if len(v) > 60 else v
                aa_disp = (aa[:60] + '..') if len(aa) > 60 else aa
                
                f.write(f"  {i+1:02d} | ORIG: {v_disp:62} -> REWRITE: {aa_disp:62} (Sim: {sim:.2f})\n")
                total_generated += 1
            
            f.write("-" * 120 + "\n")
            f.flush() # Scriviamo su disco man mano

    print(f"\n>>> SUCCESS: Exhaustive T5 Report saved to {output_file}")
    print(f"    Total samples generated across all predicates: {total_generated}")

if __name__ == "__main__":
    # Setup per riproducibilità ma con varietà
    random.seed(time.time())
    np.random.seed(int(time.time()))
    torch.manual_seed(42)
    run_t5_exhaustive_report()
