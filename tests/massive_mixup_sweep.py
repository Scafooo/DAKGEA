import sys
import torch
import logging
import random
import time
import numpy as np
import re
import os
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict
from difflib import SequenceMatcher
from rdflib import Literal
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE REPORT FINALE (MASSIVO) ---
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 10 
SWEEP_SAMPLES = 40

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'^<[^>]+>\s*', '', text).strip()

def calculate_precision_score(orig, gen, originals_set):
    """Score severo: punisce shuffling e premia novità semantica."""
    gen_l, orig_l = gen.lower().strip(), orig.lower().strip()
    if len(gen_l) < 3: return -1.0
    orig_words, gen_words = set(orig_l.split()), set(gen_l.split())
    if gen_words == orig_words and len(gen_words) > 1: return -3.0
    sim = SequenceMatcher(None, orig_l, gen_l).ratio()
    if sim > 0.98: return 0.01
    if 0.4 <= sim <= 0.9:
        score = sim * 2.0
        if gen_words - orig_words: score += 2.0
        return score
    return -0.5

def run_massive_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: DUAL-MODE OPTIMIZER (500 SAMPLES) ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=50000)
    
    print(f"    Dataset Size: {len(train_rows)} samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_large_special_v1"

    # 2. TRAINING O RESUME
    interpolator = MixupBartInterpolator(model_name="facebook/bart-large", out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting Training (Full Dataset)...")
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=1e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train(); interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)
    else:
        print(f"    [RESUME] Found existing model.")
        interpolator = MixupBartInterpolator(model_name=out_dir, out_dir=out_dir, device=device)
        interpolator.set_predicate_mapping(canonical_map)

    # 3. PREPARAZIONE TEST SETS
    print("    Extracting clean evaluation pairs...")
    test_diverse, test_similar = [], []
    aligned_by_pred = defaultdict(list)
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    
    for s_uri, t_uri in dataset.aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for p_src, v_src in s_lits.items():
            for p_tgt, v_tgt in t_lits.items():
                if canonical_map.get(p_src) == canonical_map.get(p_tgt):
                    p_tok = canonical_map[p_src]
                    if v_src.lower().strip() == v_tgt.lower().strip():
                        if len(test_similar) < SWEEP_SAMPLES: test_similar.append((p_tok, v_src, v_tgt))
                    elif len(v_src) > 4:
                        if len(test_diverse) < SWEEP_SAMPLES: test_diverse.append((p_tok, v_src, v_tgt))
                    # Popolamento per il report finale
                    aligned_by_pred[p_tok].append((p_tok, v_src, v_tgt))

    # Raccolta orfani per report
    orphan_by_pred = defaultdict(list)
    for ent in list(set(kg_src.subjects()) | set(kg_tgt.subjects()))[:2000]:
        lits = {str(p): str(o) for _, p, o in kg_src.triples((ent, None, None)) if isinstance(o, Literal)}
        lits.update({str(p): str(o) for _, p, o in kg_tgt.triples((ent, None, None)) if isinstance(o, Literal)})
        for p, v in lits.items():
            p_tok = canonical_map.get(p)
            if p_tok and len(v) > 2: orphan_by_pred[p_tok].append((p_tok, v))

    originals_set = set(clean_val(r['target']).lower() for r in train_rows)

    # 4. PHASE 2: OPTIMIZATION (VERBOSE)
    print(f"\n>>> PHASE 2: DUAL OPTIMIZATION")
    
    results_std = []
    print("    Optimizing Standard Profile...")
    for n in [0.01, 0.03, 0.05]:
        for t in [1.0, 1.3]:
            for b in [5, 8]:
                interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = n, t, b
                interpolator.similarity_threshold = 1.1 
                scs = []
                for p, v1, v2 in test_diverse:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
                    scs.append((calculate_precision_score(v1, a1, originals_set) + calculate_precision_score(v2, a2, originals_set))/2)
                avg = np.mean(scs)
                results_std.append({"n": n, "t": t, "b": b, "score": avg})
                print(f"      - N={n} T={t} B={b} -> Score: {avg:.3f}")
    best_std = sorted(results_std, key=lambda x: x['score'], reverse=True)[0]

    results_crea = []
    print("\n    Optimizing Creative Profile...")
    for n in [0.1, 0.2, 0.3]:
        for t in [1.5, 1.8, 2.2]:
            interpolator.creative_noise, interpolator.creative_temp = n, t
            interpolator.similarity_threshold = -0.1
            scs = []
            for p, v1, v2 in test_similar:
                a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
                scs.append((calculate_precision_score(v1, a1, originals_set) + calculate_precision_score(v2, a2, originals_set))/2)
            avg = np.mean(scs)
            results_crea.append({"n": n, "t": t, "score": avg})
            print(f"      - N={n} T={t} -> Score: {avg:.3f}")
    best_crea = sorted(results_crea, key=lambda x: x['score'], reverse=True)[0]

    # 5. FINAL REPORT (MASSIVO)
    print("\n" + "="*100); print(" FINAL DUAL-MODE REPORT (500 SAMPLES) ".center(100)); print("="*100)
    interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = best_std['n'], best_std['t'], best_std['b']
    interpolator.creative_noise, interpolator.creative_temp = best_crea['n'], best_crea['t']
    interpolator.similarity_threshold = 0.85

    output_file = "massive_ultimate_report_v2.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA ULTIMATE REPORT | Best STD: {best_std} | Best CREA: {best_crea}\n\n")
        
        # --- SECTION 1: ALIGNED (400 SAMPLES) ---
        f.write("SECTION 1: ALIGNED MIX-UP\n" + "-"*80 + "\n")
        count = 0
        a_preds = sorted(list(aligned_by_pred.keys()))
        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=0.5)
                    f.write(f"SAMPLE {count+1:03d} | PRED: {p_tok}\n  ORIG A: {v1}\n  AUG A': {aa}\n  ORIG B: {v2}\n  AUG B': {ab}\n")
                    f.write(f"  VOTO: [ ]/5 | NOTE: [________________________________]\n" + "-"*80 + "\n")
                    count += 1
                else: a_preds.remove(p_tok)

        # --- SECTION 2: ORPHAN (100 SAMPLES) ---
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES\n" + "-"*80 + "\n")
        o_count = 0
        o_preds = sorted(list(orphan_by_pred.keys()))
        while o_count < SAMPLES_ORPHAN and o_preds:
            for p_tok in o_preds[:]:
                if orphan_by_pred[p_tok]:
                    p_uri, v = orphan_by_pred[p_tok].pop(random.randrange(len(orphan_by_pred[p_tok])))
                    aa, _ = interpolator.interpolate_pair(v, v, predicate=p_tok, alpha=0.0)
                    f.write(f"ORPHAN {o_count+1:03d} | PRED: {p_tok}\n  ORIG: {v}\n  AUG : {aa}\n")
                    f.write(f"  VOTO: [ ]/5 | NOTE: [________________________________]\n" + "-"*80 + "\n")
                    o_count += 1
                else: o_preds.remove(p_tok)

    print(f">>> SUCCESS: Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()