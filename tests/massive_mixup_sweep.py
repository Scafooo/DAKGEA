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

# --- CONFIGURAZIONE CHIRURGICA (RTX 4090) ---
MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 96
GRAD_ACCUMULATION = 6
EPOCHS = 15
SAMPLES_ALIGNED = 400
SWEEP_SAMPLES = 40

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_human_like_score(v1, v2, gen, originals_set):
    """Calcola uno score severo: premia la SINTESI SEMANTICA, punisce il REORDERING PIGRO."""
    gen_l = gen.lower().strip()
    v1_l, v2_l = v1.lower().strip(), v2.lower().strip()
    
    if len(gen_l) < 3: return -1.0
    
    # 1. GARBAGE CHECK (Simboli tecnici)
    if len(re.findall(r'[^a-z0-9\s]', gen_l)) / (len(gen_l)+1) > 0.1:
        return -2.0
        
    # 2. PIGRIZIA CHECK (Copia o semplice reordering)
    words_gen = set(gen_l.split())
    words_orig = set(v1_l.split()) | set(v2_l.split())
    
    # Se le parole sono ESATTAMENTE le stesse (solo ordine diverso o duplicati)
    if words_gen == words_orig or words_gen.issubset(words_orig):
        return 0.05 # Punteggio bassissimo, non vogliamo questo
        
    # 3. QUALITÀ REALE (Sintesi o Parafrasi)
    new_words = words_gen - words_orig
    # Verifichiamo se le nuove parole sono sensate (non solo pezzi di parole rotte)
    valid_new_words = [w for w in new_words if len(w) > 3]
    
    sim1 = SequenceMatcher(None, v1_l, gen_l).ratio()
    sim2 = SequenceMatcher(None, v2_l, gen_l).ratio()
    
    if 0.4 <= sim1 <= 0.85 or 0.4 <= sim2 <= 0.85:
        if valid_new_words:
            return 5.0 # ECCELLENTE: ha aggiunto contesto (es. 'guitarist')
        else:
            return 2.5 # BUONO: parafrasi o fusione parziale
            
    return 0.0

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: ULTRA-PRECISION MICRO-SWEEP ".center(98) + "█")
    print("█" + " (Fractions of Noise & Alpha) ".center(98) + "█")
    print("█"*100)

    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    builder = MixupDataBuilder()
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_base_v2" # Nuova versione con meno identity

    interpolator = MixupBartInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting Training v2 (15 epochs, reduced identity)...")
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=5e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train()
        interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)

    # Preparazione test CLEAN
    aligned_test = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in list(dataset.aligned_entities):
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if vs.lower().strip() != vt.lower().strip() and len(vs) > 4 and canonical_map.get(ps) == canonical_map.get(pt):
                    aligned_test.append((canonical_map[ps], vs, vt))
        if len(aligned_test) >= SWEEP_SAMPLES: break

    # GRID SEARCH MICRO
    alphas = [0.45, 0.5] 
    noises = [0.01, 0.02]
    beams  = [5, 8]
    penalties = [1.5, 2.5] # Includiamo la penalty nello sweep
    
    results = []
    originals_set = set(clean_val(r['target']).lower() for r in train_rows)

    print(f"\n>>> MICRO-SWEEP PHASE")
    for a in alphas:
        for n in noises:
            for b in beams:
                for p in penalties:
                    interpolator.latent_noise_std = n
                    interpolator.gen_num_beams = b
                    interpolator.gen_repetition_penalty = p
                    current_scores = []
                    for pred, v1, v2 in aligned_test:
                        aug_a, aug_b = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                        s_a = calculate_human_like_score(v1, v2, aug_a, originals_set)
                        s_b = calculate_human_like_score(v1, v2, aug_b, originals_set)
                        current_scores.append((s_a + s_b) / 2)
                    
                    avg = sum(current_scores) / len(current_scores)
                    results.append({"a": a, "n": n, "b": b, "p": p, "score": avg})
                    print(f"    A={a} N={n} B={b} P={p} -> Human Score: {avg:.3f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # REPORT FINALE
    print("\n" + "="*100); print(" FINAL SOTA REPORT (DUAL AUGMENTATION) ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_repetition_penalty = best['p']
    
    output_file = "massive_base_report_v2.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA v2 REPORT | Best Config: {best}\n\n")
        
        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower().strip() != vt.lower().strip() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        report_a, count, a_preds = [], 0, list(aligned_by_pred.keys())
        while count < 400 and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aug_a, aug_b = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])
                    
                    f.write(f"{count+1:03d} | {p_tok:15} | VAL A: {v1[:20]:20} -> AUG A': {aug_a[:25]:25}\n")
                    f.write(f"    | {' ':15} | VAL B: {v2[:20]:20} -> AUG B': {aug_b[:25]:25} | Voto:[ ]/5\n")
                    f.write(f"    | {' ':15} | Note: [__________________________________________________]\n")
                    f.write("-" * 110 + "\n")
                    if count < 20: report_a.append([count+1, p_tok[:10], v1[:15], aug_a[:15], v2[:15], aug_b[:15]])
                    count += 1
                else: a_preds.remove(p_tok)
                if count >= 400: break
                
    print(tabulate(report_a, headers=["#", "PRED", "VAL A", "AUG A'", "VAL B", "AUG B'"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()