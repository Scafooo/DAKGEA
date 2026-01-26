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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE BART-BASE (BESTIA 4090) ---
MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 512       # Satura i 24GB della 4090
GRAD_ACCUMULATION = 1  # Nessun bisogno di accumulare con questo batch size
EPOCHS = 15            
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
SWEEP_SAMPLES = 50

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_precision_score(orig, gen, originals_set):
    gen_clean = gen.lower().strip()
    if len(gen_clean) < 3: return 0.0
    sim = SequenceMatcher(None, orig.lower().strip(), gen_clean).ratio()
    if sim > 0.98: return 0.05
    if 0.6 <= sim <= 0.92:
        score = sim * 1.2
        if gen_clean not in originals_set: score += 0.8
        return score
    return 0.0

def run_massive_sweep():
    print("\n" + "█"*100)
    print(f"█ {f'RTX 4090: OPTIMIZING {MODEL_NAME.upper()}'.center(96)} █")
    print("█" + " (Back to Base: Fast, Fluid, Coherent) ".center(98) + "█")
    print("█"*100)

    # 1. CARICAMENTO DATI
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    print(f"    Dataset Size: {len(train_rows)} samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_base_v1"

    # 2. TRAINING O RESUME
    model_trained = (Path(out_dir) / "pytorch_model.bin").exists() or (Path(out_dir) / "model.safetensors").exists()
    interpolator = MixupBartInterpolator(model_name=MODEL_NAME if not model_trained else out_dir, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not model_trained:
        print(f"    Starting Training ({EPOCHS} epochs)...")
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        
        args = Seq2SeqTrainingArguments(
            output_dir=out_dir, 
            per_device_train_batch_size=BATCH_SIZE, 
            num_train_epochs=EPOCHS, 
            learning_rate=5e-5, 
            fp16=True, 
            report_to="none", 
            save_strategy="no",
            dataloader_num_workers=16,
            dataloader_pin_memory=True
        )
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        print(f"    Starting Training (BESTIA MODE - 4090)...")
        trainer.train()
        interpolator.model.save_pretrained(out_dir)
        interpolator.tokenizer.save_pretrained(out_dir)
    else:
        print(f"    [RESUME] Found existing Base model.")

    # 3. TEST SUBSET CLEAN
    aligned_test = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in list(dataset.aligned_entities)[:SWEEP_SAMPLES*20]:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if vs.lower() != vt.lower() and len(vs) > 4 and canonical_map.get(ps) == canonical_map.get(pt):
                    aligned_test.append((canonical_map[ps], vs, vt))
        if len(aligned_test) >= SWEEP_SAMPLES: break

    # 4. SWEEP CHIRURGICO PER BASE
    print(f"\n>>> PHASE 2: PARAMETER OPTIMIZATION")
    alphas = [0.1, 0.2, 0.3]
    noises = [0.02, 0.05, 0.1]
    beams  = [3, 5]
    
    results = []
    originals_set = set(clean_val(r['target']).lower() for r in train_rows)

    for a in alphas:
        for n in noises:
            for b in beams:
                interpolator.latent_noise_std, interpolator.gen_num_beams = n, b
                scores = []
                for pred, v1, v2 in aligned_test:
                    res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                    scores.append(calculate_precision_score(v1, res, originals_set))
                avg_score = sum(scores) / len(scores) if scores else 0
                results.append({"a": a, "n": n, "b": b, "score": avg_score})
                print(f"    A={a} N={n} B={b} -> Score: {avg_score:.3f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    print("\n    WINNING CONFIGURATION:"); print(tabulate([best], headers="keys"))

    # 5. REPORT FINALE
    print("\n" + "="*100); print(f" ULTIMATE {MODEL_NAME.upper()} REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std, interpolator.gen_num_beams = best['n'], best['b']
    
    output_file = "massive_base_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA {MODEL_NAME} REPORT | Best Config: {best}\n\n")
        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower() != vt.lower() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        report_data, count, a_preds = [], 0, list(aligned_by_pred.keys())
        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])
                    f.write(f"{count+1:03d} | {p_tok:25} | {v1[:30]:30} | {v2[:30]:30} | {aug}\n")
                    if count < 20: report_data.append([count+1, p_tok, v1[:20], v2[:20], aug[:30]])
                    count += 1
                else: a_preds.remove(p_tok)
                if count >= SAMPLES_ALIGNED: break
                
    print(f"\n>>> SUCCESS: Base Report saved to {output_file}")
    print(tabulate(report_data, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
