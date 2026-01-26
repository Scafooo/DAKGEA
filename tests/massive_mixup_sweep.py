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

# --- CONFIGURAZIONE MICRO-CHIRURGIA (RTX 4090) ---
SAMPLES_ALIGNED = 400
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
SWEEP_SAMPLES = 30 # Ridotto leggermente per gestire 450 config in tempi umani

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

def calculate_precision_score(orig, gen, originals_set):
    """
    Score per Precisione Estrema:
    - Premia novità (non in originals_set)
    - Premia somiglianza strutturale (SequenceMatcher > 0.7)
    - Penalizza distorsioni (sim < 0.5)
    - Penalizza copie identiche (sim > 0.98)
    """
    gen_clean = gen.lower().strip()
    if len(gen_clean) < 3: return 0.0
    
    sim = SequenceMatcher(None, orig.lower().strip(), gen_clean).ratio()
    
    # 1. Copia Identica (Inutile)
    if sim > 0.98: return 0.05
    
    # 2. Sweet Spot (Nuovo ma strutturalmente coerente)
    if 0.65 <= sim <= 0.92:
        score = sim * 1.5
        if gen_clean not in originals_set:
            score += 1.0 # Forte premio per novità valida
        return score
        
    # 3. Allucinazione (Troppo diverso)
    return 0.0

def run_massive_sweep():
    print("\n" + "█"*100)
    print("█" + " RTX 4090: MICRO-PRECISION PARAMETER OPTIMIZER ".center(98) + "█")
    print("█" + " (450 Combinations: Alpha x Noise x Beams x Penalty) ".center(98) + "█")
    print("█"*100)

    # 1. DATI
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_ultimate_v5"

    # 2. MODELLO
    interpolator = MixupBartInterpolator(model_name=out_dir, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    # 3. TEST SUBSET
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

    # 4. MICRO-GRID SEARCH
    alphas = [0.05, 0.1, 0.15, 0.2, 0.3]
    noises = [0.01, 0.02, 0.03, 0.04, 0.05]
    beams  = [3, 4, 5, 6, 7, 8]
    penalties = [1.5, 1.8, 2.0]
    
    results = []
    total = len(alphas) * len(noises) * len(beams) * len(penalties)
    curr = 0
    
    originals_set = set(clean_val(r['target']).lower() for r in train_rows)

    print(f"\n>>> SCANNING {total} CONFIGURATIONS...")
    
    for a in alphas:
        for n in noises:
            for b in beams:
                for p in penalties:
                    curr += 1
                    interpolator.latent_noise_std = n
                    interpolator.gen_num_beams = b
                    interpolator.gen_repetition_penalty = p
                    
                    scores = []
                    for pred, v1, v2 in aligned_test:
                        res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                        scores.append(calculate_precision_score(v1, res, originals_set))
                    
                    avg_score = sum(scores) / len(scores) if scores else 0
                    results.append({"a": a, "n": n, "b": b, "p": p, "score": avg_score})
                    
                    if curr % 10 == 0:
                        print(f"    [{curr}/{total}] A={a} N={n} B={b} P={p} -> Score: {avg_score:.3f}", end='\r')

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    print(f"\n\n    WINNING CONFIGURATION FOUND after {total} trials:")
    print(tabulate([best], headers="keys"))

    # 5. REPORT FINALE
    print("\n" + "="*100); print(" ULTIMATE SOTA PRECISION REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std, interpolator.gen_num_beams, interpolator.gen_repetition_penalty = best['n'], best['b'], best['p']
    
    output_file = "massive_precision_optimized_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA PRECISION OPTIMIZED | Best Config: {best}\n\n")
        
        # Generazione Report Stratificato (utilizzando la logica precedente di successo)
        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower() != vt.lower() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        report_a, count, a_preds = [], 0, list(aligned_by_pred.keys())
        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])
                    f.write(f"{count+1:03d} | {p_tok:25} | {v1[:30]:30} | {v2[:30]:30} | {aug}\n")
                    if count < 20: report_a.append([count+1, p_tok, v1[:20], v2[:20], aug[:30]])
                    count += 1
                else: a_preds.remove(p_tok)
                if count >= SAMPLES_ALIGNED: break
                
    print(f"\n>>> SUCCESS: Report saved to {output_file}")
    print(tabulate(report_a, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()