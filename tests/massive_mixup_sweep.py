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
        c_score, ttype = 0.1, "lazy_permutation"
    elif new_words:
        c_score, ttype = 1.0, "semantic_leap"
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
    alphas, noises, beams, temps = [0.1, 0.3, 0.5], [0.01, 0.03, 0.05], [5, 8], [0.7, 1.0, 1.3]
    results = []
    for a in alphas:
        for n in noises:
            for b in beams:
                for t in temps:
                    interpolator.latent_noise_std, interpolator.gen_num_beams, interpolator.gen_temperature = n, b, t
                    scs = []
                    for pred, v1, v2 in aligned_test:
                        a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                        scs.append((calculate_creative_score(v1, a1, pred, format_analyzer, v2)["total"] + calculate_creative_score(v2, a2, pred, format_analyzer, v1)["total"])/2)
                    results.append({"a": a, "n": n, "b": b, "t": t, "score": np.mean(scs)})
                    print(f"    A={a} N={n} B={b} T={t} -> Score: {np.mean(scs):.3f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # REPORT FINALE MASSIVO
    print("\n" + "="*100); print(f" ULTIMATE {MODEL_NAME.upper()} REPORT ".center(100)); print("="*100)
    interpolator.latent_noise_std, interpolator.gen_num_beams, interpolator.gen_temperature = best['n'], best['b'], best['t']
    output_file = "massive_large_report_v3.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA {MODEL_NAME} REPORT | Best Config: {best}\n\n")
        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower().strip() != vt.lower().strip() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        count = 0
        a_preds = list(aligned_by_pred.keys())
        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])
                    f.write(f"{count+1:03d} | {p_tok:15} | VAL A: {v1[:20]:20} -> AUG A': {aa[:25]:25}\n")
                    f.write(f"    | {' ':15} | VAL B: {v2[:20]:20} -> AUG B': {ab[:25]:25} | Voto:[ ]/5\n")
                    f.write(f"    | {' ':15} | Note: [__________________________________________________]\n")
                    f.write("-" * 110 + "\n")
                    count += 1
                else: a_preds.remove(p_tok)
                if count >= SAMPLES_ALIGNED: break
    print(f">>> SUCCESS: Large Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()