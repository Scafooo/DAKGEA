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

# --- CONFIGURAZIONE FAST TEST ---
MODEL_NAME = "google/flan-t5-base"
BATCH_SIZE = 64
EPOCHS = 0.1 # SOLO UN DECIMO DI EPOCA PER TEST RAPIDO
TOTAL_REPORT_SAMPLES = 20 
MAX_ORPHANS_PER_PRED = 5 

torch.backends.cudnn.benchmark = True
logger = get_logger("FlanT5FastTest")
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

def run_t5_fast_test():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 FAST SMOKE TEST ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)
    
    builder = MixupDataBuilder()
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=5)
    
    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target

    # 2. COSTRUZIONE DATASET MINIMO
    print("    [1/3] Building minimum training set...")
    t5_rows = []
    src_lits_by_ent = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits_by_ent[s].append((p, str(o)))
    tgt_lits_by_ent = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits_by_ent[s].append((p, str(o)))

    for s_uri, t_uri in list(dataset.aligned_entities)[:100]: # Solo prime 100 entità
        s_attrs = src_lits_by_ent.get(s_uri, [])
        t_attrs = tgt_lits_by_ent.get(t_uri, [])
        for ps, vs in s_attrs:
            p_name = clean_p(ps, attr_map)
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    t5_rows.append({"input": f"rewrite {p_name}: {vs}", "target": vt})
                    break
    
    orphans_pool = defaultdict(list)
    for s, p, o in list(kg_src.triples((None, None, None)))[:1000]: # Solo prime 1000 triple
        if isinstance(o, Literal):
            val = str(o).strip()
            if val:
                p_name = clean_p(p, attr_map)
                orphans_pool[p_name].append(val)
    
    for p_name, vals in orphans_pool.items():
        for v in vals[:MAX_ORPHANS_PER_PRED]:
            t5_rows.append({"input": f"rewrite {p_name}: {v}", "target": v})

    random.shuffle(t5_rows)
    print(f"    Fast Dataset built: {len(t5_rows)} samples.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_flan_t5_fast_test"
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 3. TRAINING VELOCE
    print(f"    [2/3] Starting Fast Fine-tuning (0.1 Epochs)...")
    interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, force_retrain=True)

    # 4. REPORT VELOCE
    print(f"    [3/3] Generating mini-report...")
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5 FAST TEST REPORT\n")
        for i, row in enumerate(t5_rows[:TOTAL_REPORT_SAMPLES]):
            orig_val = row['input'].split(': ', 1)[1]
            p_name = row['input'].split(' ')[1].replace(':', '')
            aa, _ = interpolator.interpolate_pair(orig_val, orig_val, predicate=p_name, alpha=0.5)
            f.write(f"TEST {i+1:02d} | {p_name:20} | {orig_val[:30]} -> {aa[:30]}\n")
            
    print(f"\n>>> SUCCESS: Fast test completed. Check {output_file}")

if __name__ == "__main__":
    run_t5_fast_test()