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
BATCH_SIZE = 32        # Ridotto per BART-Large (più pesante in VRAM)
GRAD_ACCUMULATION = 8  # 32 * 8 = 256 (Batch size effettivo invariato)
FP16 = True            # Mixed precision obbligatorio per velocità
EPOCHS = 10            # Training profondo
MAX_SAMPLES = None     # Nessun limite
SWEEP_SAMPLES = 200    # Campioni per testare ogni config

# ATTIVAZIONE BOOST HARDWARE
torch.backends.cudnn.benchmark = True

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
    
    # Costruisci path assoluto
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        # Tentativo fallback comune
        alt_path = PROJECT_ROOT.parent / "data" / "raw" / "openea" / "BBC_DB"
        if alt_path.exists():
            data_path = alt_path
            logger.info(f"Found dataset at alternate path: {data_path}")
        else:
            raise FileNotFoundError(f"Could not find BBC_DB at {data_path}")

    # Verifica preliminare struttura
    if not (data_path / "knowformer_data").exists() and not (data_path / "attribute_data").exists():
        logger.warning(f"Warning: Standard subfolders not found in {data_path}. Reader might fail.")
        logger.info(f"Contents of {data_path}: {[p.name for p in data_path.iterdir()]}")

    reader = OpeneaDatasetReader()
    # Passiamo stringa assoluta
    dataset = reader.read(str(data_path))
    
    # Builder senza limiti
    builder = MixupDataBuilder(confidence_threshold=0.6)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=5000)
    
    print(f"    Dataset Size: {len(train_rows)} samples")
    if len(train_rows) > 0:
        print(f"    Sample Row: {train_rows[0]}")
    else:
        logger.error("DATASET IS EMPTY! Check path or builder.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = "./results/sweep_model_large"
    interpolator = MixupBartInterpolator(
        model_name="facebook/bart-large", 
        out_dir=out_dir,
        device=device,
        reuse_if_available=True
    )
    interpolator.set_predicate_mapping(canonical_map)

    def tokenize(batch):
        return interpolator.tokenizer(batch["input"], text_target=batch["target"], 
                                    max_length=64, truncation=True, padding="max_length")

    hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION, # Accumula gradienti
        num_train_epochs=EPOCHS,
        learning_rate=3e-5,    # LR leggermente più basso per Large
        save_strategy="no", 
        logging_steps=50,      
        report_to="none",
        fp16=FP16,
        dataloader_num_workers=8,
        dataloader_pin_memory=True, 
        dataloader_persistent_workers=True, 
    )    
    trainer = Seq2SeqTrainer(
        model=interpolator.model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model)
    )
    
    # Check if model already trained
    if (Path(out_dir) / "pytorch_model.bin").exists() or (Path(out_dir) / "model.safetensors").exists():
        print(f"    [RESUME] Found existing model in {out_dir}. Skipping training phase.")
        interpolator = MixupBartInterpolator(
            model_name=out_dir, 
            out_dir=out_dir,
            device=device
        )
        # CRITICAL: Re-register tokens from the new canonical_map
        interpolator.set_predicate_mapping(canonical_map)
    else:
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
    
    # Parametri da testare (RANGE ESPANSO PER DIVERSITÀ ESTREMA)
    alphas = [0.3, 0.5, 0.7]           # Testiamo diverse inclinazioni del mix
    noises = [0.1, 0.2, 0.3, 0.4, 0.5] # Noise spinto al limite per BART-Large
    temps  = [1.0, 1.3, 1.6, 1.8]      # Temperature elevate per creatività massima
    
    results = []
    
    # Preparazione set di test (Scansione più profonda per trovare abbastanza allineati)
    aligned_subset = []
    orphan_subset = []
    
    for row in train_rows: 
        inp = row['input']
        tgt = row['target']
        # Se contiene il separatore di coppia, è un task di allineamento
        is_aligned = ' </s> ' in inp
        
        if is_aligned and len(aligned_subset) < SWEEP_SAMPLES:
            aligned_subset.append(row)
        elif not is_aligned and len(orphan_subset) < SWEEP_SAMPLES // 2:
            orphan_subset.append(row)
            
        if len(aligned_subset) >= SWEEP_SAMPLES and len(orphan_subset) >= SWEEP_SAMPLES // 2:
            break
            
    total_configs = len(alphas) * len(noises) * len(temps)
    curr = 0
    
    print(f"    Testing {total_configs} configs on {len(aligned_subset)} aligned + {len(orphan_subset)} orphan samples...")
    
    for alpha in alphas:
        for noise in noises:
            for temp in temps:
                curr += 1
                interpolator.latent_noise_std = noise
                interpolator.gen_temperature = temp
                
                start_gen = time.time()  # <--- FIX: Added timestamp
                
                # --- TEST 1: ALIGNED MIXUP ---
                gen_aligned = []
                orig_aligned = []
                
                debug_first = True # Flag per debuggare solo il primo
                
                for row in aligned_subset:
                    # Parsing Robustezza
                    inp = row['input']
                    if ' </s> ' not in inp:
                        if debug_first: logger.warning(f"SKIP invalid format: {inp}")
                        continue
                        
                    try:
                        # Estrazione Predicato e Valori
                        # Assumiamo formato: <PRED> val1 </s> val2
                        pred_end = inp.find(' ')
                        pred = inp[:pred_end+1] # Include spazio
                        vals_part = inp[pred_end+1:]
                        vals = vals_part.split(' </s> ')
                        
                        if len(vals) < 2: 
                            if debug_first: logger.warning(f"SKIP not enough vals: {vals}")
                            continue
                            
                        v1, v2 = vals[0], vals[1]
                        
                        orig_aligned.extend([v1, v2])
                        res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred.strip(), alpha=alpha)
                        gen_aligned.append(res)
                        
                        if debug_first and curr == 1:
                            logger.info(f"[DEBUG PARSE] In: {inp}")
                            logger.info(f"[DEBUG SPLIT] Pred: '{pred}', V1: '{v1}', V2: '{v2}'")
                            logger.info(f"[DEBUG GEN]   Out: '{res}'")
                            debug_first = False
                            
                    except Exception as e:
                        logger.error(f"Error parsing row: {inp} -> {e}")
                        continue
                
                div_score = calculate_diversity_score(orig_aligned, gen_aligned)
                
                # --- TEST 2: ORPHAN VARIATION ---
                oal_score = 0
                if orphan_subset:
                    valid_variants = 0
                    for row in orphan_subset:
                        val = row['target']
                        parts = row['input'].split(' ', 1)
                        pred = parts[0]
                        # Usa alpha basso per orphan (non vogliamo mixare con il nulla)
                        # Ma sufficiente noise per generare varianti
                        res, _ = interpolator.interpolate_pair(val, val, predicate=pred, alpha=0.1)
                        
                        v_clean = val.strip().lower()
                        r_clean = res.strip().lower()
                        
                        # Premia se diverso ma simile (Variante)
                        if v_clean != r_clean and abs(len(v_clean) - len(r_clean)) < 8:
                            valid_variants += 1
                    
                    oal_score = valid_variants / len(orphan_subset)
                
                # --- GLOBAL SCORE ---
                # 70% peso su Alignment Diversity, 30% su Orphan Variation
                total_score = (div_score * 70) + (oal_score * 30)
                
                # Penalizza noise eccessivo
                if noise > 0.25: total_score *= 0.95
                
                res_dict = {
                    "alpha": alpha,
                    "noise": noise,
                    "temp": temp,
                    "div_aligned": f"{div_score:.2%}",
                    "div_orphan": f"{oal_score:.2%}",
                    "score": total_score,
                    "time": f"{time.time()-start_gen:.1f}s"
                }
                results.append(res_dict)
                print(f"    [{curr}/{total_configs}] A={alpha} N={noise} T={temp} -> Align={div_score:.2%} OAL={oal_score:.2%} Score={total_score:.2f}")

    # Ordina per Score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if not results:
        logger.error("NO VALID RESULTS COLLECTED during sweep!")
        logger.error(f"Debug Info: aligned_subset={len(aligned_subset)}, orphan_subset={len(orphan_subset)}")
        if aligned_subset:
            logger.error(f"Sample Aligned Input: {aligned_subset[0]['input']}")
        return

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

    # ---------------------------------------------------------
    # FASE 4: ORPHAN ATTRIBUTE TEST (OAL Verification)
    # ---------------------------------------------------------
    print("\n>>> PHASE 4: ORPHAN ATTRIBUTE LEARNING TEST")
    print("    Verifying if model learned to reconstruct unmatched attributes...")

    # Identifica task che sono puro denoising (input simile a target ma con noise)
    # e che non sono task di traduzione (v1 != v2). 
    # Negli orfani, input e target derivano dallo stesso valore base.
    orphan_candidates = []
    for row in train_rows:
        inp = row['input']
        tgt = row['target']
        # Euristica: se input contiene target (o viceversa) e target è lungo abbastanza
        if (tgt in inp or len(set(inp) & set(tgt)) / len(set(tgt)) > 0.8) and len(tgt) > 5:
             orphan_candidates.append(row)

    if orphan_candidates:
        print(f"    Found {len(orphan_candidates)} potential orphan/denoising tasks.")
        orphan_sample = random.sample(orphan_candidates, min(20, len(orphan_candidates)))
        
        oal_report = []
        for i, row in enumerate(orphan_sample):
            # Simula input rumoroso
            raw_input = row['input'] # Già contiene il token <PRED> e valore noise
            parts = raw_input.split(' ', 1)
            pred = parts[0]
            val = row['target']
            
            # Rigenera con il modello
            # Usiamo un po' di noise (0.1) per stimolare la variazione anche su orfani
            rec, _ = interpolator.interpolate_pair(val, val, predicate=pred, alpha=0.1)
            
            # Valutazione Sofisticata
            val_clean = val.strip().lower()
            rec_clean = rec.strip().lower()
            
            if val_clean == rec_clean:
                status = "🆗 COPY"  # Ha ricostruito bene, ma zero novità
            elif len(rec_clean) > 3 and abs(len(rec_clean) - len(val_clean)) < 10:
                # Diverso ma lunghezza simile: probabile variante valida
                status = "✅ VARIANT"
            else:
                status = "❓ CHECK" # Troppo diverso, sospetto
            
            oal_report.append([i+1, pred, val[:30], rec[:30], status])
            
        print("\n" + "="*100)
        print(" ORPHAN ATTRIBUTE VARIATION REPORT ".center(100))
        print("="*100)
        print(tabulate(oal_report, headers=["#", "PRED", "ORIGINAL", "GENERATED (OAL)", "STATUS"], tablefmt="grid"))
    else:
        print("    No orphan tasks found in sample (maybe OAL was not active in this run).")

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    run_massive_sweep()