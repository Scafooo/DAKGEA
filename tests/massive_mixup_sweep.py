import sys
import torch
import logging
import random
import time
import numpy as np
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

# --- CONFIGURAZIONE 4090 (BART-LARGE POWER) ---
BATCH_SIZE = 32        # Per BART-Large
GRAD_ACCUMULATION = 8  # Effettivo 256
FP16 = True            
EPOCHS = 10            
MAX_SAMPLES = None     
SWEEP_SAMPLES = 200    

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("MassiveSweep")

def calculate_diversity_score(original_list, generated_list):
    originals = set(s.lower().strip() for s in original_list)
    new_count = 0
    for gen in generated_list:
        if gen.lower().strip() not in originals and len(gen) > 3:
            new_count += 1
    return new_count / len(generated_list) if generated_list else 0

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: MASSIVE MIXUP SWEEP & OPTIMIZATION (BART-LARGE) ".center(98) + "█")
    print("█"*100)

    # 1. CARICAMENTO DATI
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    builder = MixupDataBuilder(confidence_threshold=0.6)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=5000)
    
    print(f"    Dataset Size: {len(train_rows)} samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_large"

    # 2. TRAINING O RESUME
    interpolator = MixupBartInterpolator(model_name="facebook/bart-large", out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if (Path(out_dir) / "pytorch_model.bin").exists() or (Path(out_dir) / "model.safetensors").exists():
        print(f"    [RESUME] Found existing model. Skipping training.")
        interpolator = MixupBartInterpolator(model_name=out_dir, out_dir=out_dir, device=device)
        interpolator.set_predicate_mapping(canonical_map)
    else:
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        training_args = Seq2SeqTrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION,
            num_train_epochs=EPOCHS, learning_rate=3e-5, save_strategy="no", report_to="none", fp16=FP16,
            dataloader_num_workers=8, dataloader_pin_memory=True, dataloader_persistent_workers=True
        )
        trainer = Seq2SeqTrainer(model=interpolator.model, args=training_args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train()
        interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)

    # 3. GRID SEARCH
    alphas = [0.3, 0.5, 0.7]
    noises = [0.1, 0.2, 0.3, 0.4, 0.5]
    temps  = [1.0, 1.3, 1.6, 1.8]
    
    aligned_subset, orphan_subset = [], []
    for row in train_rows:
        v_inp = row['input'].split(' ', 1)[1] if ' ' in row['input'] else row['input']
        v_tgt = row['target'].split(' ', 1)[1] if ' ' in row['target'] else row['target']
        # Se target non è contenuto nell'input (ignoring case), è probabilmente una traduzione
        if v_tgt.lower().strip() not in v_inp.lower().strip() and len(aligned_subset) < SWEEP_SAMPLES:
            aligned_subset.append(row)
        elif v_tgt.lower().strip() in v_inp.lower().strip() and len(orphan_subset) < SWEEP_SAMPLES // 2:
            orphan_subset.append(row)
        if len(aligned_subset) >= SWEEP_SAMPLES and len(orphan_subset) >= SWEEP_SAMPLES // 2: break

    results = []
    for alpha in alphas:
        for noise in noises:
            for temp in temps:
                interpolator.latent_noise_std, interpolator.gen_temperature = noise, temp
                start_gen = time.time()
                
                # Aligned
                gen_aligned, orig_aligned = [], []
                for row in aligned_subset:
                    parts_inp = row['input'].split(' ', 1)
                    parts_tgt = row['target'].split(' ', 1)
                    p = parts_inp[0]
                    v1 = parts_inp[1] if len(parts_inp) > 1 else ""
                    v2 = parts_tgt[1] if len(parts_tgt) > 1 else ""
                    orig_aligned.extend([v1, v2])
                    res, _ = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=alpha)
                    gen_aligned.append(res)
                div_score = calculate_diversity_score(orig_aligned, gen_aligned)
                
                # Orphan
                oal_score = 0
                if orphan_subset:
                    v_vars = 0
                    for row in orphan_subset:
                        parts_tgt = row['target'].split(' ', 1)
                        v = parts_tgt[1] if len(parts_tgt) > 1 else row['target']
                        p = row['input'].split(' ', 1)[0]
                        res, _ = interpolator.interpolate_pair(v, v, predicate=p, alpha=0.1)
                        if res.lower().strip() != v.lower().strip() and abs(len(res)-len(v)) < 8: 
                            v_vars += 1
                    oal_score = v_vars / len(orphan_subset)
                
                t_score = (div_score * 70) + (oal_score * 30)
                if noise > 0.25: t_score *= 0.95
                results.append({"alpha": alpha, "noise": noise, "temp": temp, "div_aligned": div_score, "div_orphan": oal_score, "score": t_score, "time": time.time()-start_gen})

    results.sort(key=lambda x: x['score'], reverse=True)
    best_config = results[0]
    print("\n    TOP 5 CONFIGURATIONS:"); print(tabulate(results[:5], headers="keys"))

    # 4. QUALITATIVE REPORT
    print("\n>>> PHASE 3: CHAMPION REPORT (Best Config)")
    interpolator.latent_noise_std, interpolator.gen_temperature = best_config['noise'], best_config['temp']
    report_data = []
    # Usiamo un campionamento casuale per il report qualitativo
    for i, row in enumerate(random.sample(train_rows, 50)):
        parts_inp = row['input'].split(' ', 1)
        parts_tgt = row['target'].split(' ', 1)
        p = parts_inp[0]
        v1 = parts_inp[1] if len(parts_inp) > 1 else ""
        v2 = parts_tgt[1] if len(parts_tgt) > 1 else ""
        aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best_config['alpha'])
        tag = "[✨ NEW]" if aug.lower().strip() not in [v1.lower().strip(), v2.lower().strip()] else "[⚠️ COPY]"
        report_data.append([i+1, p, v1[:25], v2[:25], f"{aug[:35]} {tag}"])
    
    print("\n" + "="*100)
    print(" QUALITATIVE REPORT (BBC_DB) ".center(100))
    print("="*100)
    print(tabulate(report_data, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
