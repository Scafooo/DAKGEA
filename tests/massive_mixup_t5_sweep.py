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

# --- CONFIGURAZIONE T5-BASE (ORIGINAL) ---
MODEL_NAME = "t5-base" # Il modello classico, robusto
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 10 
SAMPLES_ALIGNED = 200
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
    
    # Semantic Sim
    emb_orig = semantic_model.encode(orig_l, convert_to_tensor=True)
    emb_gen = semantic_model.encode(gen_l, convert_to_tensor=True)
    sim = util.cos_sim(emb_orig, emb_gen).item()
    
    if sim < 0.6: return -1.0 
    if sim > 0.98: return 0.1
    
    score = sim * 2.0
    if set(gen_l.split()) - set(orig_l.split()): score += 1.5
    return score

def run_t5_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: T5-BASE PARAPHRASE OPTIMIZER ".center(98) + "█"); print("█"*100)

    # 1. DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=50000)
    
    # Pre-process rows for T5
    t5_rows = []
    for r in train_rows:
        p_tok = r['input'].split(' ')[0]
        p_name = p_tok.replace("<", "").replace(">", "").lower()
        inp_val = r['input'].replace(p_tok, "").strip()
        tgt_val = r['target'].replace(p_tok, "").strip()
        t5_rows.append({"input": f"paraphrase {p_name}: {inp_val}", "target": tgt_val})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_t5_original_v1"

    # 2. TRAINING
    interpolator = MixupT5Interpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting T5 Training...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, force_retrain=True)
    else:
        print(f"    [RESUME] Found existing T5 model.")

    # 3. EXTRAZIONE ORPHANS & TEST SET
    print("    Extracting evaluation sets...")
    test_set = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    aligned_entities = dataset.aligned_entities
    aligned_src = {s for s, t in aligned_entities}
    aligned_tgt = {t for s, t in aligned_entities}
    
    orphan_by_pred = defaultdict(list)
    
    # Pairs (Aligned)
    for s_uri, t_uri in aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        
        # Aligned attributes
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if canonical_map.get(ps) == canonical_map.get(pt):
                    if len(vs) > 4: test_set.append((canonical_map.get(ps, ps), vs, vt))
        
        # Orphans (attr non allineati in entità allineate)
        for p, v in s_lits.items():
            found = False
            for pt in t_lits:
                if canonical_map.get(p) == canonical_map.get(pt): found = True; break
            if not found and len(v) > 4: orphan_by_pred[canonical_map.get(p, p)].append(v)
            
    # Orphans (entità non allineate)
    for s_uri, p, o in kg_src.triples():
        if s_uri not in aligned_src and isinstance(o, Literal):
            v = str(o)
            if len(v) > 4: orphan_by_pred[canonical_map.get(p, p)].append(v)

    # 4. SWEEP
    print(f"\n>>> PHASE 2: T5 PARAMETER SWEEP")
    results = []
    for a in [0.3, 0.5]:
        for n in [0.0, 0.05]:
            for t in [1.0, 1.3]: # Range più stretto per T5
                for b in [1, 5]:
                    interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = n, t, b
                    scs = []
                    for p, v1, v2 in test_set[:SWEEP_SAMPLES]:
                        a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                        scs.append((calculate_score(v1, a1) + calculate_score(v2, a2))/2)
                    avg = np.mean(scs)
                    results.append({"a": a, "n": n, "t": t, "b": b, "score": avg})
                    print(f"      - A={a} N={n} T={t} B={b} -> Score: {avg:.3f}")
    
    best = sorted(results, key=lambda x: x['score'], reverse=True)[0]

    # 5. GENERATION & REPORT
    print("\n" + "="*100); print(" GENERATING FINAL T5 REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_temperature = best['t']
    
    output_file = "massive_t5_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA T5 REPORT | Best Config: {best}\n")
        f.write("="*120 + "\n\n")
        
        # SECTION 1: ALIGNED
        f.write("SECTION 1: ALIGNED INTERPOLATIONS\n" + "-"*80 + "\n")
        count = 0
        for p, v1, v2 in test_set:
            if count >= SAMPLES_ALIGNED: break
            aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best['a'])
            sim_a = util.cos_sim(semantic_model.encode(v1), semantic_model.encode(aa)).item()
            f.write(f"PAIR {count+1:03d} | {p:15} | VAL A: {v1[:40]:40} -> AUG: {aa[:40]:40} (Sim: {sim_a:.2f}) | Voto:[ ]/5\n")
            f.write(f"         | {' ':15} | VAL B: {v2[:40]:40} -> AUG: {ab[:40]:40} | Voto:[ ]/5\n")
            f.write("-" * 120 + "\n")
            count += 1
            
        # SECTION 2: ORPHANS
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES\n" + "-"*80 + "\n")
        o_count = 0
        o_preds = sorted(list(orphan_by_pred.keys()))
        while o_count < SAMPLES_ORPHAN and o_preds:
            p = random.choice(o_preds)
            if not orphan_by_pred[p]: 
                o_preds.remove(p); continue
            v = orphan_by_pred[p].pop(random.randrange(len(orphan_by_pred[p])))
            
            # Per gli orfani, facciamo una parafrasi (alpha non conta o usiamo un metodo di parafrasi singola)
            # MixupT5Interpolator.interpolate_pair con v1=v2=v fa parafrasi
            aa, _ = interpolator.interpolate_pair(v, v, predicate=p, alpha=0.5)
            sim = util.cos_sim(semantic_model.encode(v), semantic_model.encode(aa)).item()
            f.write(f"ORPHAN {o_count+1:03d} | {p:15} | ORIG: {v[:40]:40} -> AUG: {aa[:40]:40} (Sim: {sim:.2f}) | Voto:[ ]/5\n")
            f.write("-" * 120 + "\n")
            o_count += 1

    print(f">>> SUCCESS: T5 Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_t5_sweep()
