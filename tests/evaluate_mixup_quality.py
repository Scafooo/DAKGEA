import sys
import torch
import logging
import random
import time
from pathlib import Path
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
from difflib import SequenceMatcher

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

logging.basicConfig(level=logging.INFO)

def string_sim(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def run_massive_quality_test():
    print("\n" + "█"*100)
    print("█" + " MASSIVE ROBUST MIX-UP EVALUATION (300 SAMPLES) ".center(98) + "█")
    print("█"*100)

    # 1. Load Dataset
    reader = OpeneaDatasetReader()
    dataset = reader.read("data/raw/openea/BBC_DB")
    
    # 2. Build Training Data (Robust Translation)
    builder = MixupDataBuilder(confidence_threshold=0.6)
    # Limita il numero di righe totali per un test più rapido ma significativo
    train_rows, canonical_map = builder.build_training_data(dataset)
    if len(train_rows) > 30000:
        train_rows = random.sample(train_rows, 30000)
    
    # 3. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    interpolator = MixupBartInterpolator(model_name="facebook/bart-base", device=device, reuse_if_available=False)
    interpolator.set_predicate_mapping(canonical_map)
    
    # Parametri bilanciati basati sugli sweep precedenti
    interpolator.latent_noise_std = 0.25 
    interpolator.gen_temperature = 1.3
    interpolator.gen_repetition_penalty = 1.2

    # 4. Fine-tuning
    def tokenize(batch):
        return interpolator.tokenizer(batch["input"], text_target=batch["target"], 
                                    max_length=96, truncation=True, padding="max_length")

    hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
    
    print(f"\n--- Training: {len(hf_ds)} campioni (Balanced Strategy) ---")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/mixup_massive_test",
        per_device_train_batch_size=16,
        num_train_epochs=3, # Ridotto a 3 per velocità
        report_to="none",
        fp16=torch.cuda.is_available(),
        logging_steps=100
    )
    
    trainer = Seq2SeqTrainer(
        model=interpolator.model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model)
    )
    trainer.train()

    # 5. Massive Augmentation Test (300 samples)
    print("\n" + "="*100)
    print(" REPORT: MASSIVE AUGMENTATION RESULTS ".center(100))
    print("=" * 100)

    # Campioniamo 300 righe dal dataset di training (valori reali)
    # Nota: usiamo righe che hanno coppie diverse per vedere la creatività
    test_samples = [r for r in train_rows if r['input'] != r['target']][:300]
    if len(test_samples) < 300: test_samples = train_rows[:300]

    results = []
    print(f"\n{'#':<4} | {'CONCEPT':<15} | {'IN A':<20} | {'IN B':<20} | {'AUGMENTED'}")
    print("-" * 100)

    for i, sample in enumerate(test_samples):
        p_tok = sample['input'].split(' ', 1)[0]
        v_a = sample['input'].split(' ', 1)[1]
        v_b = sample['target'].split(' ', 1)[1]
        
        try:
            res, _ = interpolator.interpolate_pair(v_a, v_b, predicate=p_tok, alpha=0.5)
            
            # Analisi Qualitativa
            is_identical = res.lower().strip() == v_a.lower().strip() or res.lower().strip() == v_b.lower().strip()
            similarity = max(string_sim(res, v_a), string_sim(res, v_b))
            
            results.append({
                'p': p_tok, 'a': v_a, 'b': v_b, 'out': res, 
                'identical': is_identical, 'sim': similarity
            })
            
            if i < 50: # Stampiamo solo i primi 50 per brevità ma calcoliamo su 300
                marker = "⚠️ COPY" if is_identical else "✨ NEW"
                print(f"{i+1:<4} | {p_tok:<15} | {v_a[:20]:<20} | {v_b[:20]:<20} | {res} [{marker}]")
        except: continue

    # Statistiche Finali
    total = len(results)
    new_values = sum(1 for r in results if not r['identical'])
    avg_sim = sum(r['sim'] for r in results) / total
    
    print("\n" + "█"*100)
    print(f" FINAL STATS (N={total}) ".center(100))
    print("█"*100)
    print(f"  • Diversità (Valori Nuovi): {new_values}/{total} ({100*new_values/total:.1f}%)")
    print(f"  • Remembrance (Similarità Media): {avg_sim:.4f}")
    print(f"  • Fallimenti (Copie): {total - new_values}")
    print("█"*100)

if __name__ == "__main__":
    run_massive_quality_test()
