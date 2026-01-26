import sys
import torch
import logging
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

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE OPTIMIZER TOTALE (RTX 4090) ---
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 30
SWEEP_SAMPLES = 40

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_sota_score(original_list, generated_list):
    """Calcola la qualità totale: Coerenza (70%) + Diversità (30%)."""
    originals = set(s.lower().strip() for s in original_list)
    valid_variants = 0
    garbage = 0
    
    for gen in generated_list:
        clean = gen.lower().strip()
        # Check Garbage (simboli, lunghezza, loop)
        if len(clean) < 2 or len(re.findall(r'[^a-zA-Z0-9\s]', clean)) / (len(clean)+1) > 0.3:
            garbage += 1; continue
        if any(clean.count(w) > 3 for w in clean.split() if len(w) > 2):
            garbage += 1; continue
        # Check Novelty
        if clean not in originals: valid_variants += 1
            
    diversity = valid_variants / len(generated_list) if generated_list else 0
    purity = 1.0 - (garbage / len(generated_list))
    return (purity * 0.7) + (diversity * 0.3)

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: FULL COMBINATORIAL PARAMETER OPTIMIZER ".center(98) + "█")
    print("█" + " (Alpha x Noise x Beams x Temp x Penalty) ".center(98) + "█")
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

    if not model_trained:
        print(f"    [TRAIN] Model v5 not found. Training...")
        interpolator.fine_tune(train_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=3e-5, force_retrain=True)

    # 3. PREPARAZIONE TEST SUBSET (Solo valori PULITI)
    aligned_test = []
    # Usiamo un set per evitare duplicati causati dalle diverse varianti di noise nel training
    seen_pairs = set()
    
    for row in train_rows:
        # Nel nostro builder, il target contiene sempre il valore PULITO <PRED> Valore
        # L'input invece contiene il rumore. Noi vogliamo testare con il PULITO.
        # Troviamo le coppie di traduzione guardando i target di righe correlate
        pass 

    # Approccio più semplice: Estraiamo i valori puliti dai target delle righe di training
    # che rappresentano una traduzione (v_inp_clean != v_tgt_clean)
    for row in train_rows:
        # v_tgt è sempre pulito. 
        v_tgt = clean_val(row['target'])
        pred = row['target'].split(' ')[0]
        
        # Per ricostruire la coppia (A, B) pulita, cerchiamo nel dataset 
        # una riga dove quel v_tgt era l'input (ma il builder mette noise nell'input...)
        # Quindi facciamo così: usiamo il target della riga corrente come VAL B
        # e cerchiamo un VAL A plausibile.
        
        # In realtà, il modo più corretto è rigenerare le coppie pulite dal dataset originale
        # ma per lo sweep possiamo semplicemente simulare: VAL A = VAL B (Denoising)
        # o VAL A = VAL B_variante.
        
    # CORREZIONE DEFINITIVA: Usiamo direttamente il dataset originale per il test
    print("    Extracting clean evaluation pairs from dataset...")
    aligned_test = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in list(dataset.aligned_entities)[:SWEEP_SAMPLES*5]:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        
        for p_src, v_src in s_lits.items():
            for p_tgt, v_tgt in t_lits.items():
                if v_src.lower() != v_tgt.lower() and len(v_src) > 3:
                    # Se i predicati sono mappati nello stesso token, è una coppia valida
                    if canonical_map.get(p_src) == canonical_map.get(p_tgt):
                        pred_tok = canonical_map[p_src]
                        aligned_test.append((pred_tok, v_src, v_tgt))
        if len(aligned_test) >= SWEEP_SAMPLES: break

    # 4. FULL GRID SEARCH
    print(f"\n>>> PHASE 2: FULL GRID SEARCH (32 CONFIGS)")
    
    alphas = [0.3, 0.5]
    noises = [0.05, 0.1]
    beams = [1, 5]
    temps = [1.0, 1.3]
    penalties = [1.2, 1.6]
    
    results = []
    total = len(alphas) * len(noises) * len(beams) * len(temps) * len(penalties)
    curr = 0
    
    for a in alphas:
        for n in noises:
            for b in beams:
                for t in temps:
                    for p in penalties:
                        curr += 1
                        # Configurazione al volo
                        interpolator.latent_noise_std = n
                        interpolator.gen_num_beams = b
                        interpolator.gen_temperature = t
                        interpolator.gen_repetition_penalty = p
                        interpolator.gen_top_k = 50
                        
                        gen_list, orig_list = [], []
                        for pred, v1, v2 in aligned_test:
                            res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                            gen_list.append(res); orig_list.extend([v1, v2])
                        
                        score = calculate_sota_score(orig_list, gen_list)
                        results.append({"a": a, "n": n, "b": b, "t": t, "p": p, "score": score})
                        print(f"    [{curr}/{total}] A={a} N={n} B={b} T={t} P={p} -> Score: {score:.2f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    print("\n    WINNING CONFIGURATION:"); print(tabulate([best], headers="keys"))

    # 5. REPORT FINALE MASSIVO
    print("\n" + "="*100); print(" ULTIMATE SOTA REPORT (STRATIFIED) ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_temperature = best['t']
    interpolator.gen_repetition_penalty = best['p']
    
    # Generazione report come richiesto (stratificato)
    aligned_by_pred = defaultdict(list)
    for row in train_rows:
        v_inp, v_tgt = clean_val(row['input']), clean_val(row['target'])
        pred = row['input'].split(' ')[0]
        if v_inp.lower() != v_tgt.lower() and len(v_inp) > 3:
            aligned_by_pred[pred].append((pred, v_inp, v_tgt))

    output_file = "massive_ultimate_report_v2.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA ULTIMATE REPORT | Best Config: {best}\n\n")
        report_a, count, a_preds = [], 0, list(aligned_by_pred.keys())
        while count < SAMPLES_ALIGNED and a_preds:
            for pred in a_preds[:]:
                if aligned_by_pred[pred]:
                    p_uri, v1, v2 = aligned_by_pred[pred].pop(random.randrange(len(aligned_by_pred[pred])))
                    aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_uri, alpha=best['a'])
                    f.write(f"{count+1:03d} | {p_uri:25} | {v1[:30]:30} | {v2[:30]:30} | {aug}\n")
                    if count < 20: report_a.append([count+1, p_uri, v1[:20], v2[:20], aug[:30]])
                    count += 1
                else: a_preds.remove(pred)
                if count >= SAMPLES_ALIGNED: break
                
    print(f"\n>>> SUCCESS: Ultimate v2 Report saved to {output_file}")
    print(tabulate(report_a, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()