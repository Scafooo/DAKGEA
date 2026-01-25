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

# --- CONFIGURAZIONE 4090 ---
BATCH_SIZE = 64        # Alto per sfruttare i 24GB VRAM
FP16 = True            # Mixed precision per velocità
EPOCHS = 10            # Training profondo
MAX_SAMPLES = None     # Nessun limite, dataset completo
SWEEP_SAMPLES = 200    # Campioni per testare ogni config nello sweep

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("MassiveSweep")

def calculate_diversity_score(original_list, generated_list):
    """Calcola quanto sono nuove le stringhe generate rispetto alle originali."""
    originals = set(original_list)
    new_count = 0
    for gen in generated_list:
        if gen not in originals and len(gen) > 3:
            new_count += 1
    return new_count / len(generated_list) if generated_list else 0

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: MASSIVE MIXUP SWEEP & OPTIMIZATION ".center(98) + "█")
    print("█" + " (Fine-tuning -> Grid Search -> Best Config Report) ".center(98) + "█")
    print("█"*100)

    # ---------------------------------------------------------
    # FASE 1: PREPARAZIONE DATI & FINE-TUNING
    # ---------------------------------------------------------
    print("\n>>> PHASE 1: FULL DATASET FINE-TUNING")
    
    reader = OpeneaDatasetReader()
    dataset = reader.read("data/raw/openea/BBC_DB")
    
    # Builder senza limiti
    builder = MixupDataBuilder(confidence_threshold=0.6)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=5000)
    
    print(f"    Dataset Size: {len(train_rows)} samples")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = "./results/sweep_model"
    interpolator = MixupBartInterpolator(
        model_name="facebook/bart-base", 
        out_dir=out_dir,
        device=device,
        reuse_if_available=True  # Usa True se vuoi saltare il training se esiste già
    )
    interpolator.set_predicate_mapping(canonical_map)

    def tokenize(batch):
        return interpolator.tokenizer(batch["input"], text_target=batch["target"], 
                                    max_length=64, truncation=True, padding="max_length")

    hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=5e-5,
        save_strategy="no", 
        logging_steps=500,
        report_to="none",
        fp16=FP16,
        dataloader_num_workers=4
    )
    
    trainer = Seq2SeqTrainer(
        model=interpolator.model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model)
    )
    
    print(f"    Starting Training ({EPOCHS} epochs)...")
    t0 = time.time()
    trainer.train()
    print(f"    Training Complete in {time.time()-t0:.1f}s")
    interpolator.model.save_pretrained(out_dir)
    interpolator.tokenizer.save_pretrained(out_dir)

    # ---------------------------------------------------------
    # FASE 2: HYPERPARAMETER SWEEP
    # ---------------------------------------------------------
    print("\n>>> PHASE 2: GRID SEARCH SWEEP")
    
    # Parametri da testare
    alphas = [0.1, 0.3, 0.5]
    noises = [0.0, 0.1, 0.2, 0.3]
    temps  = [1.0, 1.2, 1.4]
    
    results = []
    
    # Preparazione set di test (presi dal training per velocità ma rappresentativi)
    test_subset = train_rows[:SWEEP_SAMPLES]
    
    total_configs = len(alphas) * len(noises) * len(temps)
    curr = 0
    
    print(f"    Testing {total_configs} configurations on {SWEEP_SAMPLES} samples each...")
    
    for alpha in alphas:
        for noise in noises:
            for temp in temps:
                curr += 1
                interpolator.latent_noise_std = noise
                interpolator.gen_temperature = temp
                
                generated = []
                originals = []
                
                start_gen = time.time()
                for row in test_subset:
                    # Estrai input
                    inp = row['input'] # Formato: <PRED> <VAL1> <VAL2>
                    parts = inp.split('>', 1)
                    if len(parts) < 2: continue
                    pred = parts[0] + '>'
                    vals = parts[1].strip().split(' </s> ')
                    if len(vals) < 2: continue
                    v1, v2 = vals[0], vals[1]
                    
                    originals.append(v1)
                    originals.append(v2)
                    
                    # Genera
                    res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=alpha)
                    generated.append(res)
                
                # Calcola metriche
                diversity = calculate_diversity_score(originals, generated)
                
                # Score euristico: Vogliamo Alta Diversità ma non Garbage.
                # Assumiamo che se diversity > 90% forse è garbage, se < 10% è copia.
                # Target ideale: 60-80% diversity con noise controllato.
                
                score = diversity * 100 
                # Penalizza noise troppo alto se non porta benefici
                if noise > 0.25: score *= 0.95 
                
                res_dict = {
                    "alpha": alpha,
                    "noise": noise,
                    "temp": temp,
                    "diversity": f"{diversity:.2%}",
                    "score": score,
                    "time": f"{time.time()-start_gen:.1f}s"
                }
                results.append(res_dict)
                print(f"    [{curr}/{total_configs}] A={alpha} N={noise} T={temp} -> Div={diversity:.2%} Score={score:.2f}")

    # Ordina per Score
    results.sort(key=lambda x: x['score'], reverse=True)
    best_config = results[0]
    
    print("\n    TOP 5 CONFIGURATIONS:")
    print(tabulate(results[:5], headers="keys", tablefmt="simple"))

    # ---------------------------------------------------------
    # FASE 3: REPORT FINALE (BEST CONFIG)
    # ---------------------------------------------------------
    print("\n>>> PHASE 3: CHAMPION REPORT (Best Config)")
    
    interpolator.latent_noise_std = best_config['noise']
    interpolator.gen_temperature = best_config['temp']
    best_alpha = best_config['alpha']
    
    print(f"    Running qualitative analysis with: Noise={best_config['noise']}, Temp={best_config['temp']}, Alpha={best_alpha}")
    
    # Genera tabella qualitativa su 50 esempi random
    report_data = []
    eval_subset = random.sample(train_rows, 50)
    
    for i, row in enumerate(eval_subset):
        inp = row['input']
        parts = inp.split('>', 1)
        pred = parts[0] + '>'
        vals = parts[1].strip().split(' </s> ')
        v1, v2 = vals[0], vals[1]
        
        aug, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=best_alpha)
        
        # Check if new
        is_new = aug != v1 and aug != v2
        tag = "[✨ NEW]" if is_new else "[⚠️ COPY]"
        
        report_data.append([i+1, pred, v1[:20], v2[:20], f"{aug[:30]} {tag}"])

    print("\n" + "="*100)
    print(" QUALITATIVE REPORT (BBC_DB) ".center(100))
    print("="*100)
    print(tabulate(report_data, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))
    
    # Salva config suggerita
    print(f"\n[SUGGESTION] Update your config/augmentation/mixup_plm.yaml with:")
    print(f"  latent_noise_std: {best_config['noise']}")
    print(f"  temperature: {best_config['temp']}")
    print(f"  base_alpha: {best_config['alpha']}")

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    run_massive_sweep()