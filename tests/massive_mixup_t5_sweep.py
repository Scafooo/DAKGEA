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

# --- CONFIGURAZIONE BILANCIATA ---
MODEL_NAME = "google/flan-t5-base"
BATCH_SIZE = 32
EPOCHS = 3 # 3 epoche sono ottimali per fine-tuning su dataset medi/grandi
TOTAL_REPORT_SAMPLES = 400 
MAX_ORPHANS_PER_PRED = 500 # Limita la dominanza di certi predicati nel training

torch.backends.cudnn.benchmark = True
logger = get_logger("FlanT5Exhaustive")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_attr_names(dataset_path):
    attr_map = {}
    for i in [1, 2]:
        path = Path(dataset_path) / f"attribute_data/attr_names{i}"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        attr_map[parts[0].strip()] = parts[1].strip()
    return attr_map

def clean_p(uri, attr_map):
    uri_str = str(uri)
    if uri_str in attr_map:
        return attr_map[uri_str].replace(' ', '_').lower()
    return uri_str.split('/')[-1].split('#')[-1].replace('>', '').replace('<', '').lower()

def run_t5_balanced_training():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 BALANCED TRAINING (STRATIFIED KG) ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)
    
    builder = MixupDataBuilder()
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=10)
    
    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target

    # 2. COSTRUZIONE DATASET BILANCIATO
    print("    [1/3] Building balanced training set...")
    t5_rows = []
    
    # Pre-indexing
    src_lits_by_ent = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits_by_ent[s].append((p, str(o)))
    tgt_lits_by_ent = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits_by_ent[s].append((p, str(o)))

    # A. COPPIE ALLINEATE (Alta priorità)
    aligned_count = 0
    for s_uri, t_uri in dataset.aligned_entities:
        s_attrs = src_lits_by_ent.get(s_uri, [])
        t_attrs = tgt_lits_by_ent.get(t_uri, [])
        for ps, vs in s_attrs:
            p_name = clean_p(ps, attr_map)
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    t5_rows.append({"input": f"rewrite {p_name}: {vs}", "target": vt})
                    t5_rows.append({"input": f"rewrite {p_name}: {vt}", "target": vs})
                    aligned_count += 2
                    break
    
    # B. ORFANI STRATIFICATI (Per coprire ID, Date, etc. senza eccedere)
    # Raggruppiamo orfani per predicato
    orphans_pool = defaultdict(list)
    for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
        if isinstance(o, Literal):
            val = str(o).strip()
            if val:
                p_name = clean_p(p, attr_map)
                orphans_pool[p_name].append(val)
    
    orphan_count = 0
    for p_name, vals in orphans_pool.items():
        unique_vals = list(set(vals))
        random.shuffle(unique_vals)
        # Ne prendiamo solo un tot per predicato
        selected = unique_vals[:MAX_ORPHANS_PER_PRED]
        for v in selected:
            t5_rows.append({"input": f"rewrite {p_name}: {v}", "target": v})
            orphan_count += 1

    random.shuffle(t5_rows)
    print(f"    Dataset built: Aligned={aligned_count}, Orphans={orphan_count} (Total={len(t5_rows)})")
    print(f"    Predicates covered: {len(orphans_pool)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_flan_t5_balanced_v1"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 3. TRAINING
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    [2/3] Starting Balanced Fine-tuning (3 Epochs)...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, force_retrain=True)
    else:
        print(f"    [2/3] Balanced model already exists.")

    # 4. REPORT
    print(f"    [3/3] Generating report...")
    # Usiamo Round-Robin per il report per vedere tutti i predicati
    report_buckets = defaultdict(deque)
    for row in t5_rows:
        p_name = row['input'].split(' ')[1].replace(':', '')
        val = row['input'].split(': ', 1)[1]
        report_buckets[p_name].append(val)
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5 BALANCED REPORT\n")
        f.write(f"Training: Stratified KG (Max {MAX_ORPHANS_PER_PRED} per pred). Epochs: {EPOCHS}\n")
        f.write("="*120 + "\n\n")
        
        gen_count = 0
        p_names = sorted(list(report_buckets.keys()))
        while gen_count < TOTAL_REPORT_SAMPLES and p_names:
            for p in p_names[:]:
                if not report_buckets[p]:
                    p_names.remove(p); continue
                
                orig_val = report_buckets[p].popleft()
                aa, _ = interpolator.interpolate_pair(orig_val, orig_val, predicate=p, alpha=0.5)
                sim = util.cos_sim(semantic_model.encode(orig_val), semantic_model.encode(aa)).item()
                
                f.write(f"SAMPLE {gen_count+1:03d} | {p:20} | ORIG: {orig_val[:45]:45} -> REWRITE: {aa[:45]:45} (Sim: {sim:.2f})\n")
                
                gen_count += 1
                if gen_count >= TOTAL_REPORT_SAMPLES: break
            
    print(f"\n>>> SUCCESS: Report saved to {output_file}")

if __name__ == "__main__":
    run_t5_balanced_training()