import sys
import torch
import random
import time
import numpy as np
import re
import os
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE TRAINING ULTIMATE (30 EPOCHS) ---
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 30 # Massima profondità di apprendimento

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: ULTIMATE TRAINING & AUGMENTATION (BART-LARGE v5) ".center(98) + "█")
    print("█" + " (30 Epochs + Cosine Decay + Label Smoothing + Structural Noise) ".center(98) + "█")
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
        print(f"    [RESUME] Found existing ultimate model v5.")
        best = {"alpha": 0.5, "noise": 0.2, "temp": 1.5} # Parametri creativi definitivi
    else:
        print(f"    Starting ULTIMATE TRAINING v5 (30 epochs)...")
        # In questo run usiamo il builder con Structural Noise (già impostato prima)
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        
        # L'interpolatore ora usa Cosine Scheduler e Label Smoothing grazie alla modifica precedente
        interpolator.fine_tune(train_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-5, force_retrain=True)
        best = {"alpha": 0.5, "noise": 0.2, "temp": 1.5}

    # 3. GENERAZIONE 500 CAMPIONI STRATIFICATI
    interpolator.latent_noise_std, interpolator.gen_temperature = best['noise'], best['temp']
    
    aligned_by_pred = defaultdict(list)
    orphan_by_pred = defaultdict(list)
    
    for row in train_rows:
        v_inp, v_tgt = clean_val(row['input']), clean_val(row['target'])
        pred = row['input'].split(' ')[0]
        if v_inp.lower() != v_tgt.lower() and len(v_inp) > 3:
            aligned_by_pred[pred].append((pred, v_inp, v_tgt))
        else:
            orphan_by_pred[pred].append((pred, v_tgt))

    output_file = "massive_ultimate_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(" DAKGEA: ULTIMATE AUGMENTATION REPORT (BART-LARGE v5) \n")
        f.write(f" Depth: 30 Epochs | Features: Cosine Decay, Label Smoothing, Robust Noise \n")
        f.write("================================================================================\n\n")

        # Sezione 1: Aligned
        f.write("SECTION 1: CREATIVE ALIGNED MIX-UP (N=400)\n")
        f.write("-" * 80 + "\n")
        report_a, count, a_preds = [], 0, list(aligned_by_pred.keys())
        while count < SAMPLES_ALIGNED and a_preds:
            for p in a_preds[:]:
                if aligned_by_pred[p]:
                    p_uri, v1, v2 = aligned_by_pred[p].pop(random.randrange(len(aligned_by_pred[p])))
                    aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_uri, alpha=best['alpha'])
                    tag = "[CREATIVE]" if aug.lower() not in [v1.lower(), v2.lower()] else "[STABLE]"
                    f.write(f"{count+1:03d} | {p_uri:25} | {v1[:30]:30} | {v2[:30]:30} | {aug} {tag}\n")
                    if count < 20: report_a.append([count+1, p_uri, v1[:20], v2[:20], aug[:30]])
                    count += 1
                else: a_preds.remove(p)
                if count >= SAMPLES_ALIGNED: break

        # Sezione 2: Orphan
        f.write("\n\nSECTION 2: CREATIVE ORPHAN VARIANTS (N=100)\n")
        f.write("-" * 80 + "\n")
        o_count = 0
        o_preds = list(orphan_by_pred.keys())
        while o_count < SAMPLES_ORPHAN and o_preds:
            for p in o_preds[:]:
                if orphan_by_pred[p]:
                    p_uri, v = orphan_by_pred[p].pop(random.randrange(len(orphan_by_pred[p])))
                    aug, _ = interpolator.interpolate_pair(v, v, predicate=p_uri, alpha=0.1)
                    tag = "[VAR]" if aug.lower() != v.lower() else "[COPY]"
                    f.write(f"{o_count+1:03d} | {p_uri:25} | {v[:60]:60} | {aug} {tag}\n")
                    o_count += 1
                else: o_preds.remove(p)
                if o_count >= SAMPLES_ORPHAN: break

    print(f"\n>>> SUCCESS: Ultimate Report saved to {output_file}")
    print("\nPreview of first 20 Samples:")
    print(tabulate(report_a, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
