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
from difflib import SequenceMatcher
from rdflib import Literal
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_t5_interpolator import MixupT5Interpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE FLAN-T5 (RTX 4090) ---
MODEL_NAME = "google/flan-t5-base" # Instruction-tuned: molto più smart
BATCH_SIZE = 32        # Flan-base è gestibile
GRAD_ACCUMULATION = 8
EPOCHS = 10            # Converge velocemente
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
SWEEP_SAMPLES = 40


torch.backends.cudnn.benchmark = True
logger = get_logger("T5Sweep")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_val(text):
    return re.sub(r'^<[^>]+>\s*', '', text).strip()

def calculate_score(orig, gen):
    gen_l, orig_l = gen.lower().strip(), orig.lower().strip()
    if len(gen_l) < 3: return -1.0
    
    # Check T5 artifacts
    if "extra_id" in gen_l: return -2.0
    
    # Check Shuffling
    if set(gen_l.split()) == set(orig_l.split()) and len(gen_l.split()) > 1: return -3.0
    
    # Semantic Sim
    emb_orig = semantic_model.encode(orig, convert_to_tensor=True)
    emb_gen = semantic_model.encode(gen_l, convert_to_tensor=True)
    sim = util.cos_sim(emb_orig, emb_gen).item()
    
    if sim < 0.6: return -1.0 
    if sim > 0.98: return 0.1
    
    score = sim * 2.0
    if set(gen_l.split()) - set(orig.lower().split()): score += 1.5
    return score

def run_t5_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: T5 MIXUP OPTIMIZER ".center(98) + "█"); print("█"*100)

    # 1. DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=50000)
    
    # Converti rows per T5: "<NAME> val" -> "augment name: val"
    t5_rows = []
    for r in train_rows:
        p_tok = r['input'].split(' ')[0] # <NAME>
        p_name = p_tok.replace("<", "").replace(">", "").lower()
        
        inp_val = r['input'].replace(p_tok, "").strip()
        tgt_val = r['target'].replace(p_tok, "").strip()
        
        t5_rows.append({
            "input": f"augment {p_name}: {inp_val}",
            "target": tgt_val
        })
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_t5_v1"

    # 2. TRAINING
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting T5 Training...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, force_retrain=True)
    else:
        print(f"    [RESUME] Found existing T5 model.")

    # 3. TEST SETS
    print("    Extracting clean evaluation pairs...")
    test_set = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in dataset.aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if canonical_map.get(ps) == canonical_map.get(pt):
                    if len(vs) > 4:
                        test_set.append((canonical_map[ps], vs, vt))
        if len(test_set) >= SWEEP_SAMPLES * 2: break

    # 4. SWEEP
    print(f"\n>>> PHASE 2: T5 PARAMETER SWEEP")
    results = []
    # T5 risponde diversamente a noise/temp, usiamo range più ampi
    for a in [0.3, 0.5]: # Aggiunto sweep su Alpha
        for n in [0.05, 0.1, 0.2]:
            for t in [1.0, 1.5]:
                for b in [3, 5]:
                    interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = n, t, b
                    scs = []
                    for p, v1, v2 in test_set[:SWEEP_SAMPLES]:
                        a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                        scs.append((calculate_score(v1, a1) + calculate_score(v2, a2))/2)
                    avg = np.mean(scs)
                    results.append({"a": a, "n": n, "t": t, "b": b, "score": avg})
                    print(f"      - A={a} N={n} T={t} B={b} -> Score: {avg:.3f}")
    
    best = sorted(results, key=lambda x: x['score'], reverse=True)[0]

    # 5. REPORT
    print("\n" + "="*100); print(" FINAL T5 REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_temperature = best['t']
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA T5 REPORT | Best: {best}\n\n")
        count = 0
        for p_tok, v1, v2 in test_set:
            if count >= SAMPLES_ALIGNED: break
            aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])
            sim_a = util.cos_sim(semantic_model.encode(v1), semantic_model.encode(aa)).item()
            f.write(f"{count+1:03d} | {p_tok:15} | VAL A: {v1[:30]:30} -> AUG A': {aa[:35]:35} (Sim: {sim_a:.2f})\n")
            f.write(f"    | {' ':15} | VAL B: {v2[:30]:30} -> AUG B': {ab[:35]:35} | Voto:[ ]/5\n")
            f.write("-" * 110 + "\n")
            count += 1

    print(f">>> SUCCESS: T5 Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_t5_sweep()
