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

# --- CONFIGURAZIONE DUAL OPTIMIZER (RTX 4090) ---
MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 96
GRAD_ACCUMULATION = 6
EPOCHS = 15
SWEEP_SAMPLES = 40

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_precision_score(orig, gen, originals_set):
    """Score severo: punisce shuffling e premia novità semantica."""
    gen_l, orig_l = gen.lower().strip(), orig.lower().strip()
    if len(gen_l) < 3: return -1.0
    
    orig_words, gen_words = set(orig_l.split()), set(gen_l.split())
    if gen_words == orig_words and len(gen_words) > 1: return -3.0 # Molto severo con lo shuffle
    
    sim = SequenceMatcher(None, orig_l, gen_l).ratio()
    if sim > 0.98: return 0.01 # Quasi copia
    
    if 0.4 <= sim <= 0.9:
        score = sim * 2.0
        if gen_words - orig_words: score += 2.0 # Bonus parola nuova
        return score
    return -0.5

def run_massive_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: DUAL-MODE PARAMETER OPTIMIZER ".center(98) + "█"); print("█"*100)

    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder()
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_base_v2"

    interpolator = MixupBartInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    # TRAINING SE NECESSARIO
    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print("    Training..."); hf_ds = HFDataset.from_list(train_rows).map(lambda b: interpolator.tokenizer(b["input"], text_target=b["target"], max_length=64, truncation=True, padding="max_length"), batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=5e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train(); interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)

    # PREPARAZIONE TEST (DIVERSI vs IDENTICI)
    test_diverse, test_similar = [], []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in dataset.aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if canonical_map.get(ps) == canonical_map.get(pt):
                    if vs.lower().strip() == vt.lower().strip():
                        if len(test_similar) < SWEEP_SAMPLES: test_similar.append((canonical_map[ps], vs, vt))
                    elif len(vs) > 4:
                        if len(test_diverse) < SWEEP_SAMPLES: test_diverse.append((canonical_map[ps], vs, vt))
        if len(test_diverse) >= SWEEP_SAMPLES and len(test_similar) >= SWEEP_SAMPLES: break

    originals_set = set(clean_val(r['target']).lower() for r in train_rows)

    # --- PHASE 2: DUAL OPTIMIZATION ---
    print(f"\n>>> PHASE 2: DUAL OPTIMIZATION")
    
    # 1. Ottimizza Profilo Standard (per coppie diverse)
    print("    Optimizing Standard Profile...")
    std_results = []
    for n in [0.01, 0.03, 0.05]:
        for t in [1.0, 1.2, 1.4]:
            for b in [3, 5, 8]:
                interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = n, t, b
                interpolator.similarity_threshold = 1.1 # Forza Standard mode
                scs = []
                for p, v1, v2 in test_diverse:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
                    scs.append((calculate_precision_score(v1, a1, originals_set) + calculate_precision_score(v2, a2, originals_set))/2)
                std_results.append({"n": n, "t": t, "b": b, "score": np.mean(scs)})
    
    best_std = sorted(std_results, key=lambda x: x['score'], reverse=True)[0]
    print(f"    BEST STANDARD: N={best_std['n']} T={best_std['t']} B={best_std['b']} (Score: {best_std['score']:.3f})")

    # 2. Ottimizza Profilo Creative (per coppie identiche)
    print("\n    Optimizing Creative Profile (Identical Pairs)...")
    crea_results = []
    # RANGE AGGRESSIVO per forzare la diversità
    for n in [0.1, 0.2, 0.3, 0.4]: # Noise molto alto
        for t in [1.5, 1.8, 2.2]: # Temperature estreme
            for p_pen in [2.5, 3.5, 4.5]: # Penalità durissima
                interpolator.creative_noise, interpolator.creative_temp, interpolator.creative_penalty = n, t, p_pen
                interpolator.similarity_threshold = -0.1 # Forza creative
                scs = []
                for p, v1, v2 in test_similar:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
                    # Score che premia SOLO se ha cambiato qualcosa (novità semantica)
                    scs.append((calculate_precision_score(v1, a1, originals_set) + calculate_precision_score(v2, a2, originals_set))/2)
                crea_results.append({"n": n, "t": t, "p": p_pen, "score": np.mean(scs)})
    
    best_crea = sorted(crea_results, key=lambda x: x['score'], reverse=True)[0]
    print(f"    BEST CREATIVE: N={best_crea['n']} T={best_crea['t']} P={best_crea['p']} (Score: {best_crea['score']:.3f})")

    # --- PHASE 3: FINAL REPORT ---
    print("\n" + "="*100); print(" ULTIMATE DUAL-MODE SOTA REPORT ".center(100)); print("="*100)
    
    interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = best_std['n'], best_std['t'], best_std['b']
    interpolator.creative_noise, interpolator.creative_temp, interpolator.creative_penalty = best_crea['n'], best_crea['t'], best_crea['p']
    interpolator.similarity_threshold = 0.85

    output_file = "massive_base_report_v5.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA v5 DUAL-MODE REPORT\nSTD: {best_std} | CREA: {best_crea}\n\n")
        aligned_test_full = test_diverse[:25] + test_similar[:25]
        for i, (p, v1, v2) in enumerate(aligned_test_full):
            aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
            f.write(f"{i+1:03d} | {p:15} | VAL A: {v1}\n    | {' ':15} | VAL B: {v2}\n")
            f.write(f"    | {' ':15} | AUG A': {aa}\n    | {' ':15} | AUG B': {ab}\n")
            f.write(f"    | {' ':15} | Voto:[ ]/5 | Note: [________________________________]\n")
            f.write("-" * 100 + "\n")
            
    print(f">>> SUCCESS: Dual Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()