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

# --- CONFIGURAZIONE BART-LARGE (POTENZA MASSIMA) ---
MODEL_NAME = "facebook/bart-large"
BATCH_SIZE = 32        # VRAM Safe per Large
GRAD_ACCUMULATION = 8  # Effettivo 256
EPOCHS = 10            # Il Large converge prima
SAMPLES_ALIGNED = 400
SWEEP_SAMPLES = 40

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

class PredicateFormatAnalyzer:
    def __init__(self, training_rows):
        self.stats = defaultdict(lambda: {"token_counts": [], "char_lengths": [], "has_digits": [], "samples": set()})
        self._analyze(training_rows)
        self._compute_stats()

    def _analyze(self, rows):
        for row in rows:
            match = re.match(r'(<[^>]+>)\s*(.+)', row['target'])
            if match:
                pred, val = match.groups()
                self.stats[pred]["token_counts"].append(len(val.split()))
                self.stats[pred]["char_lengths"].append(len(val))
                self.stats[pred]["has_digits"].append(1.0 if re.search(r'\d', val) else 0.0)
                self.stats[pred]["samples"].add(val.lower().strip())

    def _compute_stats(self):
        for pred, s in self.stats.items():
            if s["token_counts"]:
                s["expected_tokens"] = (np.percentile(s["token_counts"], 5), np.percentile(s["token_counts"], 95))
                s["expected_length"] = (np.percentile(s["char_lengths"], 5), np.percentile(s["char_lengths"], 95))
                s["digit_ratio"] = np.mean(s["has_digits"])

    def format_score(self, pred, gen):
        s = self.stats.get(pred)
        if not s or not gen.strip(): return 0.0
        score, tokens, length = 1.0, len(gen.split()), len(gen)
        if not (s["expected_tokens"][0] <= tokens <= s["expected_tokens"][1]): score *= 0.7
        if not (s["expected_length"][0] <= length <= s["expected_length"][1]): score *= 0.7
        return score

def calculate_creative_score(orig, gen, pred, analyzer, other=None):
    gen_c, orig_c = gen.strip(), orig.strip()
    other_c = other.strip() if other else ""
    if len(gen_c) < 3: return {"total": 0, "transform": "garbage"}
    
    f_score = analyzer.format_score(pred, gen_c)
    
    # LEXICAL NOVELTY SEVERA
    gen_words = set(re.findall(r'\w+', gen_c.lower()))
    orig_words = set(re.findall(r'\w+', orig_c.lower()))
    other_words = set(re.findall(r'\w+', other_c.lower()))
    source_words = orig_words | other_words
    new_words = [w for w in (gen_words - source_words) if len(w) > 2]
    
    sim_orig = SequenceMatcher(None, orig_c.lower(), gen_c.lower()).ratio()
    
    if sim_orig > 0.95:
        c_score, ttype = 0.05, "identity"
    elif gen_words == source_words or gen_words.issubset(source_words):
        c_score, ttype = -5.0, "lazy_permutation" # PUNIZIONE NUCLEARE CONTRO LO SHUFFLING
    elif new_words:
        c_score, ttype = 1.5, "semantic_leap" # PREMIO MAGGIORE PER VERA NOVITÀ
    elif 0.5 <= sim_orig <= 0.85:
        c_score, ttype = 0.7, "partial_merge"
    else:
        c_score, ttype = 0.2, "garbage"
    
    return {"total": (f_score * 0.3) + (c_score * 0.7), "transform": ttype}

def run_massive_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: BART-LARGE ULTIMATE PRECISION ".center(98) + "█"); print("█"*100)

    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder()
    train_rows, canonical_map = builder.build_training_data(dataset)
    format_analyzer = PredicateFormatAnalyzer(train_rows)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_large_v3" # Nuova cartella per non mischiare i pesi

    interpolator = MixupBartInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print(f"    Starting Training ({MODEL_NAME})..."); hf_ds = HFDataset.from_list(train_rows).map(lambda b: interpolator.tokenizer(b["input"], text_target=b["target"], max_length=64, truncation=True, padding="max_length"), batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=2e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train(); interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)

    print("    Extracting clean evaluation pairs...")
    aligned_test = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in list(dataset.aligned_entities):
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if vs.lower().strip() != vt.lower().strip() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                    aligned_test.append((canonical_map[ps], vs, vt))
        if len(aligned_test) >= SWEEP_SAMPLES: break

    # GRID SEARCH CHIRURGICO PER LARGE
    print(f"\n>>> PHASE 2: PARAMETER OPTIMIZATION")
    alphas = [0.1, 0.3, 0.5]
    noises = [0.01, 0.03, 0.05]
    beams = [3, 5, 8]
    temps = [0.7, 1.0, 1.3]
    penalties = [1.2, 1.5, 2.0] # Aggiunta la ricerca della penalty
    
    results = []
    for a in alphas:
        for n in noises:
            for b in beams:
                for t in temps:
                    for p in penalties:
                        interpolator.latent_noise_std = n
                        interpolator.gen_num_beams = b
                        interpolator.gen_temperature = t
                        interpolator.gen_repetition_penalty = p
                        
                        scs = []
                        for pred, v1, v2 in aligned_test:
                            a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                            scs.append((calculate_creative_score(v1, a1, pred, format_analyzer, v2)["total"] + 
                                        calculate_creative_score(v2, a2, pred, format_analyzer, v1)["total"])/2)
                        
                        avg_sc = np.mean(scs)
                        results.append({"a": a, "n": n, "b": b, "t": t, "p": p, "score": avg_sc})
                        
                        # Log ogni tanto per non intasare
                        if len(results) % 10 == 0:
                            print(f"    Trial {len(results)}: A={a} N={n} B={b} T={t} P={p} -> Score: {avg_sc:.3f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # 5. REPORT FINALE MASSIVO (STRATIFICATO COMPLETO)
    print("\n" + "="*100); print(" ULTIMATE SOTA REPORT (ALL ATTRIBUTES - CLEAN) ".center(100)); print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_temperature = best['t']
    interpolator.gen_repetition_penalty = best['p']
    
    # 1. RACCOLTA DATI ALLINEATI
    aligned_by_pred = defaultdict(list)
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    for s_uri, t_uri in dataset.aligned_entities:
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                if vs.lower().strip() != vt.lower().strip() and len(vs) > 3:
                    if canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

    # 2. RACCOLTA DATI ORFANI (Date, ID, etc.)
    orphan_by_pred = defaultdict(list)
    all_entities = set(kg_src.subjects()) | set(kg_tgt.subjects())
    for ent in list(all_entities)[:2000]: # Campiona entità per trovare orfani
        lits = {str(p): str(o) for _, p, o in kg_src.triples((ent, None, None)) if isinstance(o, Literal)}
        lits.update({str(p): str(o) for _, p, o in kg_tgt.triples((ent, None, None)) if isinstance(o, Literal)})
        for p, v in lits.items():
            p_tok = canonical_map.get(p)
            if p_tok and len(v) > 2:
                # Se questo predicato non appare spesso negli allineamenti, è un buon candidato orfano
                orphan_by_pred[p_tok].append((p_tok, v))

    output_file = "massive_base_report_v4.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA v4 ULTIMATE REPORT | Best Config: {best}\n\n")
        
        # --- SEZIONE 1: MIX-UP ALLINEATO (300 campioni) ---
        f.write("SECTION 1: ALIGNED MIX-UP\n" + "-"*100 + "\n")
        count = 0
        a_preds = sorted(list(aligned_by_pred.keys()))
        while count < 300 and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])
                    
                    # FORMATO FULL: Niente tagli, tutto su righe dedicate
                    f.write(f"SAMPLE {count+1:03d} | PRED: {p_tok}\n")
                    f.write(f"  ORIG A: {v1}\n")
                    f.write(f"  AUG A': {aa}\n")
                    f.write(f"  ORIG B: {v2}\n")
                    f.write(f"  AUG B': {ab}\n")
                    f.write(f"  VOTO: [ ]/5 | NOTE: [__________________________________________________]\n")
                    f.write("-" * 100 + "\n")
                    count += 1
                else: a_preds.remove(p_tok)

        # --- SEZIONE 2: ORPHAN AUGMENTATION (100 campioni) ---
        f.write("\n\nSECTION 2: ORPHAN ATTRIBUTE AUGMENTATION\n" + "-"*100 + "\n")
        o_count = 0
        o_preds = sorted(list(orphan_by_pred.keys()))
        while o_count < 100 and o_preds:
            for p_tok in o_preds[:]:
                if orphan_by_pred[p_tok]:
                    p_uri, v = orphan_by_pred[p_tok].pop(random.randrange(len(orphan_by_pred[p_tok])))
                    aa, _ = interpolator.interpolate_pair(v, v, predicate=p_tok, alpha=0.0)
                    
                    f.write(f"ORPHAN {o_count+1:03d} | PRED: {p_tok}\n")
                    f.write(f"  ORIG: {v}\n")
                    f.write(f"  AUG : {aa}\n")
                    f.write(f"  VOTO: [ ]/5 | NOTE: [__________________________________________________]\n")
                    f.write("-" * 100 + "\n")
                    o_count += 1
                else: o_preds.remove(p_tok)
    print(f">>> SUCCESS: Large Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()