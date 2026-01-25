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

# --- CONFIGURAZIONE PER TEST LOCALE (VELOCE) ---
BATCH_SIZE = 8
GRAD_ACCUMULATION = 1
FP16 = False           
EPOCHS = 1            
MAX_SAMPLES = 50
SWEEP_SAMPLES = 5    

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("MassiveSweep")

# Parametri Sweep ridotti
alphas = [0.5]
noises = [0.2]
temps  = [1.2]

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
    print(f"    Canonical Map size: {len(canonical_map)}")
    
    # FORZIAMO BART-BASE PER IL TEST LOCALE E SALTIAMO IL TRAINING
    interpolator = MixupBartInterpolator(model_name="facebook/bart-base", out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)
    
    # Simuliamo il training caricando il modello base direttamente
    print("    [DEBUG] Skipping training for local logic verification...")


    # 3. GRID SEARCH
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
                print(f"    - Testing Config: A={alpha} N={noise} T={temp}")
                for idx, row in enumerate(aligned_subset):
                    parts_inp = row['input'].split(' ', 1)
                    parts_tgt = row['target'].split(' ', 1)
                    p = parts_inp[0]
                    v1 = parts_inp[1] if len(parts_inp) > 1 else ""
                    v2 = parts_tgt[1] if len(parts_tgt) > 1 else ""
                    orig_aligned.extend([v1, v2])
                    
                    if idx % 50 == 0:
                        print(f"      [Progress] Aligned: {idx}/{len(aligned_subset)}", end='\r')
                    
                    res, _ = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=alpha)
                    gen_aligned.append(res)
                print(f"      [Progress] Aligned: Done")
                div_score = calculate_diversity_score(orig_aligned, gen_aligned)
                
                # Orphan
                oal_score = 0
                if orphan_subset:
                    v_vars = 0
                    for idx, row in enumerate(orphan_subset):
                        parts_tgt = row['target'].split(' ', 1)
                        v = parts_tgt[1] if len(parts_tgt) > 1 else row['target']
                        p = row['input'].split(' ', 1)[0]
                        
                        if idx % 25 == 0:
                            print(f"      [Progress] Orphan: {idx}/{len(orphan_subset)}", end='\r')
                            
                        res, _ = interpolator.interpolate_pair(v, v, predicate=p, alpha=0.1)
                        if res.lower().strip() != v.lower().strip() and abs(len(res)-len(v)) < 8: 
                            v_vars += 1
                    print(f"      [Progress] Orphan: Done")
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
