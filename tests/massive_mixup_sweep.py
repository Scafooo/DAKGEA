import sys
import torch
import logging
import random
import time
import numpy as np
import re
from pathlib import Path
from tabulate import tabulate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE PRODUZIONE 4090 ---
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 10
SWEEP_SAMPLES = 100

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("MassiveSweep")

def clean_val(text):
    """Estrae il valore rimuovendo il predicato <PRED>."""
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_diversity_score(original_list, generated_list):
    originals = set(s.lower().strip() for s in original_list)
    new_count = sum(1 for gen in generated_list if gen.lower().strip() not in originals and len(gen) > 3)
    return new_count / len(generated_list) if generated_list else 0

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: ULTIMATE COHERENT MIXUP SWEEP (BART-LARGE) ".center(98) + "█")
    print("█" + " (Strict Pairing + Orphan Attribute Learning) ".center(98) + "█")
    print("█"*100)

    # 1. CARICAMENTO DATI (Con Coherent Pairing)
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    # Threshold 0.3 per evitare "Nomi + Date"
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    print(f"    Dataset Size: {len(train_rows)} samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_stable_v3"

    # 2. TRAINING
    interpolator = MixupBartInterpolator(model_name="facebook/bart-large", out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting Training (Coherent Mode)...")
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=3e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train()
        interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)
    else:
        print(f"    [RESUME] Using existing coherent model.")
        interpolator = MixupBartInterpolator(model_name=out_dir, out_dir=out_dir, device=device)
        interpolator.set_predicate_mapping(canonical_map)

    # 3. PREPARAZIONE SUBSET DI TEST (Strict)
    aligned_test, orphan_test = [], []
    for row in train_rows:
        v_inp, v_tgt = clean_val(row['input']), clean_val(row['target'])
        pred = row['input'].split(' ')[0]
        if v_inp.lower() != v_tgt.lower() and len(v_inp) > 3 and len(v_tgt) > 3:
            if len(aligned_test) < SWEEP_SAMPLES: aligned_test.append((pred, v_inp, v_tgt))
        else:
            if len(orphan_test) < SWEEP_SAMPLES // 2: orphan_test.append((pred, v_tgt))
        if len(aligned_test) >= SWEEP_SAMPLES and len(orphan_test) >= SWEEP_SAMPLES // 2: break

    # 4. GRID SEARCH (70/30 Score)
    alphas = [0.3, 0.5]
    noises = [0.0, 0.1, 0.2]
    temps  = [1.0, 1.3]
    
    print(f"\n>>> PHASE 2: GRID SEARCH SWEEP")
    results = []
    for a in alphas:
        for n in noises:
            for t in temps:
                interpolator.latent_noise_std, interpolator.gen_temperature = n, t
                # Test Aligned
                gen_a, orig_a = [], []
                for p, v1, v2 in aligned_test:
                    res, _ = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                    gen_a.append(res); orig_a.extend([v1, v2])
                div_score = calculate_diversity_score(orig_a, gen_a)
                # Test Orphan
                oal_v = 0
                for p, v in orphan_test:
                    res, _ = interpolator.interpolate_pair(v, v, predicate=p, alpha=0.1)
                    if res.lower() != v.lower() and abs(len(res)-len(v)) < 8: oal_v += 1
                oal_score = oal_v / len(orphan_test) if orphan_test else 0
                
                total = (div_score * 70) + (oal_score * 30)
                results.append({"alpha": a, "noise": n, "temp": t, "align": div_score, "oal": oal_score, "score": total})
                print(f"    A={a} N={n} T={t} -> Score: {total:.2f} (Align: {div_score:.1%}, OAL: {oal_score:.1%})")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # 5. REPORT QUALITATIVO FINALE (DIVERSIFICATO & STRATIFICATO)
    print("\n" + "="*100); print(" CHAMPION QUALITATIVE REPORT (BBC_DB) - DIVERSIFIED ".center(100)); print("="*100)
    interpolator.latent_noise_std, interpolator.gen_temperature = best['noise'], best['temp']
    
    # Raggruppiamo gli allineamenti per predicato per campionamento stratificato
    from collections import defaultdict
    aligned_by_pred = defaultdict(list)
    for p, v1, v2 in aligned_test:
        # Preferiamo esempi NON banali (v1 != v2 sostanzialmente)
        if len(set(v1.split()) ^ set(v2.split())) > 0:
            aligned_by_pred[p].append((p, v1, v2))
    
    # Sezione 1: ALIGNED MIXUP (75 esempi stratificati)
    print("\n[SECTION 1: STRATIFIED ALIGNED MIX-UP (N=75)]")
    report_a = []
    preds = list(aligned_by_pred.keys())
    count = 0
    while count < 75 and preds:
        for p in preds[:]: # Iteriamo sui predicati
            if aligned_by_pred[p]:
                p_uri, v1, v2 = aligned_by_pred[p].pop(0)
                aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_uri, alpha=best['alpha'])
                tag = "[✨]" if aug.lower() not in [v1.lower(), v2.lower()] else ""
                report_a.append([count+1, p_uri, v1[:25], v2[:25], f"{aug[:35]} {tag}"])
                count += 1
            else:
                preds.remove(p)
            if count >= 75: break

    print(tabulate(report_a, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

    # Sezione 2: ORPHAN VARIATION (25 esempi stratificati, incluse date)
    print("\n[SECTION 2: STRATIFIED ORPHAN VARIATION (N=25)]")
    orphan_by_pred = defaultdict(list)
    for p, v in orphan_test:
        orphan_by_pred[p].append((p, v))
        
    report_o = []
    o_preds = list(orphan_by_pred.keys())
    o_count = 0
    while o_count < 25 and o_preds:
        for p in o_preds[:]:
            if orphan_by_pred[p]:
                p_uri, v = orphan_by_pred[p].pop(0)
                aug, _ = interpolator.interpolate_pair(v, v, predicate=p_uri, alpha=0.1)
                tag = "[✨]" if aug.lower() != v.lower() else ""
                report_o.append([o_count+1, p_uri, v[:50], f"{aug[:50]} {tag}"])
                o_count += 1
            else:
                o_preds.remove(p)
            if o_count >= 25: break
            
    print(tabulate(report_o, headers=["#", "PRED", "ORIGINAL VALUE", "GENERATED VARIANT"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
