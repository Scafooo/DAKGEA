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
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE OPTIMIZER (RTX 4090) ---
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 30
SWEEP_SAMPLES = 40 # Campioni per config (ridotto per testare più combinazioni)

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_sota_score(original_list, generated_list):
    """Calcola la qualità totale: Diversità Semantica - (Hallucination + Garbage)."""
    originals = set(s.lower().strip() for s in original_list)
    valid_variants = 0
    garbage = 0
    
    for gen in generated_list:
        clean = gen.lower().strip()
        # 1. Check Garbage (simboli strani, troppo corti, ripetizioni)
        if len(clean) < 2 or len(re.findall(r'[^a-zA-Z0-9\s]', clean)) / (len(clean)+1) > 0.3:
            garbage += 1
            continue
        # 2. Check Loops
        if any(clean.count(w) > 3 for w in clean.split() if len(w) > 2):
            garbage += 1
            continue
        # 3. Check Novelty
        if clean not in originals:
            valid_variants += 1
            
    diversity = valid_variants / len(generated_list) if generated_list else 0
    purity = 1.0 - (garbage / len(generated_list))
    
    # Lo score premia la purezza (coerenza) al 70% e la diversità al 30%
    return (purity * 0.7) + (diversity * 0.3)

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: SOTA PARAMETER OPTIMIZER (BART-LARGE v5) ".center(98) + "█")
    print("█" + " (Beams, Top-K, Penalty, Noise Search) ".center(98) + "█")
    print("█"*100)

    # 1. CARICAMENTO DATI
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_ultimate_v5"

    # 2. TRAINING O RESUME
    model_trained = (Path(out_dir) / "pytorch_model.bin").exists() or (Path(out_dir) / "model.safetensors").exists()
    interpolator = MixupBartInterpolator(model_name="facebook/bart-large" if not model_trained else out_dir, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if model_trained:
        print(f"    [RESUME] Model v5 found. Ready for optimization.")
    else:
        print(f"    [TRAIN] Model v5 not found. Training 30 epochs...")
        interpolator.fine_tune(train_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-5, force_retrain=True)

    # 3. PREPARAZIONE TEST SUBSET
    aligned_test = []
    for row in train_rows:
        v_inp, v_tgt = clean_val(row['input']), clean_val(row['target'])
        pred = row['input'].split(' ')[0]
        if v_inp.lower() != v_tgt.lower() and len(v_inp) > 3:
            if len(aligned_test) < SWEEP_SAMPLES: aligned_test.append((pred, v_inp, v_tgt))
        if len(aligned_test) >= SWEEP_SAMPLES: break

    # 4. SUPER GRID SEARCH
    print(f"\n>>> PHASE 2: SOTA GRID SEARCH")
    
    # Parametri da ottimizzare
    noises = [0.05, 0.1]
    beams = [1, 5]           # 1=Sampling, 5=Beam Search
    top_ks = [50]            # Filtro base anti-garbage
    penalties = [1.2, 1.6]   # Freno a mano
    
    results = []
    total = len(noises) * len(beams) * len(penalties)
    curr = 0
    
    for n in noises:
        for b in beams:
            for p in penalties:
                curr += 1
                interpolator.latent_noise_std = n
                interpolator.gen_num_beams = b
                interpolator.gen_repetition_penalty = p
                interpolator.gen_top_k = 50
                interpolator.gen_temperature = 1.2 if b == 1 else 1.0
                
                gen_a, orig_a = [], []
                for pred, v1, v2 in aligned_test:
                    res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=0.5)
                    gen_a.append(res); orig_a.extend([v1, v2])
                
                q_score = calculate_sota_score(orig_a, gen_a)
                res_dict = {"noise": n, "beams": b, "pen": p, "score": q_score}
                results.append(res_dict)
                print(f"    [{curr}/{total}] Beam={b} Noise={n} Pen={p} -> SOTA Score: {q_score:.2f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    
    print("\n    WINNING CONFIGURATION:")
    print(tabulate([best], headers="keys"))

    # 5. FINAL QUALITATIVE REPORT
    print("\n" + "="*100); print(" ULTIMATE SOTA QUALITATIVE REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['noise']
    interpolator.gen_num_beams = best['beams']
    interpolator.gen_repetition_penalty = best['pen']
    
    aligned_by_pred = defaultdict(list)
    orphan_by_pred = defaultdict(list)
    for row in train_rows:
        v_inp, v_tgt = clean_val(row['input']), clean_val(row['target'])
        pred = row['input'].split(' ')[0]
        if v_inp.lower() != v_tgt.lower() and len(v_inp) > 3:
            aligned_by_pred[pred].append((pred, v_inp, v_tgt))
        else:
            orphan_by_pred[pred].append((pred, v_tgt))

    output_file = "massive_sota_optimized_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA SOTA OPTIMIZED REPORT | Best Config: {best}\n\n")
        
        # Section 1: Aligned
        report_a, count, a_preds = [], 0, list(aligned_by_pred.keys())
        while count < SAMPLES_ALIGNED and a_preds:
            for p in a_preds[:]:
                if aligned_by_pred[p]:
                    p_uri, v1, v2 = aligned_by_pred[p].pop(random.randrange(len(aligned_by_pred[p])))
                    aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_uri, alpha=0.5)
                    line = f"{count+1:03d} | {p_uri:25} | {v1[:30]:30} | {v2[:30]:30} | {aug}\n"
                    f.write(line)
                    if count < 20: report_a.append([count+1, p_uri, v1[:20], v2[:20], aug[:30]])
                    count += 1
                else: a_preds.remove(p)
                if count >= SAMPLES_ALIGNED: break
                
    print(f"\n>>> SUCCESS: Optimized Report saved to {output_file}")
    print(tabulate(report_a, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
