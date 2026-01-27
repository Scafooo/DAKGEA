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
from difflib import SequenceMatcher
from rdflib import Literal
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# --- CONFIGURAZIONE SOTA CON SEMANTIC CHECK ---
MODEL_NAME = "facebook/bart-large"
BATCH_SIZE = 32
GRAD_ACCUMULATION = 8
EPOCHS = 10 
SWEEP_SAMPLES = 40
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

# Modello Semantico per Validazione
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_val(text):
    return re.sub(r'^<[^>]+>\s*', '', text).strip()

def calculate_semantic_score(orig, gen, originals_set):
    gen_l = gen.lower().strip()
    if len(gen_l) < 3: return -1.0
    
    # 1. GARBAGE CHECK (Simboli)
    if len(re.findall(r'[^a-z0-9\s]', gen_l)) / (len(gen_l)+1) > 0.1: return -2.0
    
    # 2. PIGRIZIA CHECK (Shuffling)
    orig_words, gen_words = set(orig.lower().split()), set(gen_l.split())
    if gen_words == orig_words and len(gen_words) > 1: return -3.0
    
    # 3. SEMANTIC SIMILARITY CHECK (Il Giudice Reale)
    emb_orig = semantic_model.encode(orig, convert_to_tensor=True)
    emb_gen = semantic_model.encode(gen_l, convert_to_tensor=True)
    sem_sim = util.cos_sim(emb_orig, emb_gen).item()
    
    # Logica di punteggio
    if sem_sim < 0.6: return -1.0 # Troppo diverso (Garbage o Allucinazione)
    if sem_sim > 0.98: return 0.1 # Troppo uguale (Noioso)
    
    score = sem_sim * 2.0
    if gen_words - orig_words: score += 1.5 # Bonus Lexical Novelty
    
    return score

def run_massive_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: SOTA SEMANTIC OPTIMIZER ".center(98) + "█"); print("█"*100)

    # 1. CARICAMENTO DATI
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder(confidence_threshold=0.6, value_match_threshold=0.3)
    train_rows, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=50000)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_large_special_v1"

    # 2. TRAINING
    interpolator = MixupBartInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting Training...")
        def tokenize(batch):
            return interpolator.tokenizer(batch["input"], text_target=batch["target"], max_length=64, truncation=True, padding="max_length")
        hf_ds = HFDataset.from_list(train_rows).map(tokenize, batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=1e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train(); interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)
    else:
        print(f"    [RESUME] Found existing model.")
        interpolator = MixupBartInterpolator(model_name=out_dir, out_dir=out_dir, device=device)
        interpolator.set_predicate_mapping(canonical_map)

    # 3. TEST SETS
    print("    Extracting clean evaluation pairs...")
    test_diverse, test_similar = [], []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in dataset.aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for p_src, v_src in s_lits.items():
            for p_tgt, v_tgt in t_lits.items():
                if canonical_map.get(p_src) == canonical_map.get(p_tgt):
                    p_tok = canonical_map[p_src]
                    if v_src.lower().strip() == v_tgt.lower().strip():
                        if len(test_similar) < SWEEP_SAMPLES: test_similar.append((p_tok, v_src, v_tgt))
                    elif len(v_src) > 4:
                        if len(test_diverse) < SWEEP_SAMPLES: test_diverse.append((p_tok, v_src, v_tgt))
        if len(test_diverse) >= SWEEP_SAMPLES and len(test_similar) >= SWEEP_SAMPLES: break

    originals_set = set(clean_val(r['target']).lower() for r in train_rows)

    # 4. SEMANTIC SWEEP
    print(f"\n>>> PHASE 2: SEMANTIC DUAL OPTIMIZATION")
    
    # 1. Standard (Diverse)
    print("    Optimizing Standard Profile...")
    results_std = []
    for n in [0.01, 0.03, 0.05]:
        for t in [1.0, 1.2]: 
            for b in [5, 8]:
                interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = n, t, b
                interpolator.similarity_threshold = 1.1 
                scs = []
                for p, v1, v2 in test_diverse:
                    a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
                    scs.append((calculate_semantic_score(v1, a1, originals_set) + calculate_semantic_score(v2, a2, originals_set))/2)
                avg = np.mean(scs)
                results_std.append({"n": n, "t": t, "b": b, "score": avg})
                print(f"      - N={n} T={t} B={b} -> SemScore: {avg:.3f}")
    best_std = sorted(results_std, key=lambda x: x['score'], reverse=True)[0]

    # 2. Creative (Similar)
    print("\n    Optimizing Creative Profile...")
    results_crea = []
    for n in [0.05, 0.1]:
        for t in [1.3, 1.5]: 
            interpolator.creative_noise, interpolator.creative_temp = n, t
            interpolator.similarity_threshold = -0.1
            scs = []
            for p, v1, v2 in test_similar:
                a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=0.5)
                scs.append((calculate_semantic_score(v1, a1, originals_set) + calculate_semantic_score(v2, a2, originals_set))/2)
            avg = np.mean(scs)
            results_crea.append({"n": n, "t": t, "score": avg})
            print(f"      - N={n} T={t} -> SemScore: {avg:.3f}")
    best_crea = sorted(results_crea, key=lambda x: x['score'], reverse=True)[0]

    # 5. REPORT
    print("\n" + "="*100); print(" FINAL SEMANTIC REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std, interpolator.gen_temperature, interpolator.gen_num_beams = best_std['n'], best_std['t'], best_std['b']
    interpolator.creative_noise, interpolator.creative_temp = best_crea['n'], best_crea['t']
    interpolator.similarity_threshold = 0.85

    output_file = "massive_semantic_report_v1.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA SEMANTIC REPORT | Best STD: {best_std} | Best CREA: {best_crea}\n\n")
        
        # Aligned Report
        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower().strip() != vt.lower().strip() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        count = 0
        a_preds = sorted(list(aligned_by_pred.keys()))
        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=0.5)
                    
                    # Calcola similarità semantica per il report
                    sim_a = util.cos_sim(semantic_model.encode(v1), semantic_model.encode(aa)).item()
                    sim_b = util.cos_sim(semantic_model.encode(v2), semantic_model.encode(ab)).item()
                    
                    f.write(f"{count+1:03d} | {p_tok:15} | VAL A: {v1[:30]:30} -> AUG A': {aa[:35]:35} (Sim: {sim_a:.2f})\n")
                    f.write(f"    | {' ':15} | VAL B: {v2[:30]:30} -> AUG B': {ab[:35]:35} (Sim: {sim_b:.2f}) | Voto:[ ]/5\n")
                    f.write("-" * 110 + "\n")
                    count += 1
                else: a_preds.remove(p_tok)

        # Orphan Report
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTES\n" + "-"*80 + "\n")
        orphan_by_pred = defaultdict(list)
        for ent in list(set(kg_src.subjects()) | set(kg_tgt.subjects()))[:2000]:
            lits = {str(p): str(o) for _, p, o in kg_src.triples((ent, None, None)) if isinstance(o, Literal)}
            lits.update({str(p): str(o) for _, p, o in kg_tgt.triples((ent, None, None)) if isinstance(o, Literal)})
            for p, v in lits.items():
                p_tok = canonical_map.get(p)
                if p_tok and len(v) > 2: orphan_by_pred[p_tok].append((p_tok, v))
        
        o_count = 0
        o_preds = sorted(list(orphan_by_pred.keys()))
        while o_count < SAMPLES_ORPHAN and o_preds:
            for p_tok in o_preds[:]:
                if orphan_by_pred[p_tok]:
                    p_uri, v = orphan_by_pred[p_tok].pop(random.randrange(len(orphan_by_pred[p_tok])))
                    aa, _ = interpolator.interpolate_pair(v, v, predicate=p_tok, alpha=0.0)
                    sim = util.cos_sim(semantic_model.encode(v), semantic_model.encode(aa)).item()
                    f.write(f"ORPHAN {o_count+1:03d} | {p_tok:15} | ORIG: {v[:40]:40} -> AUG: {aa[:40]:40} (Sim: {sim:.2f}) | Voto:[ ]/5\n")
                    o_count += 1
                else: o_preds.remove(p_tok)

    print(f">>> SUCCESS: Semantic Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()