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

# --- CONFIGURAZIONE BART-BASE (OTTIMIZZATA PER RTX 4090 - 24GB VRAM) ---
MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 128       # RTX 4090 può gestire batch grandi
GRAD_ACCUMULATION = 4  # Effective batch = 512
EPOCHS = 15
SAMPLES_ALIGNED = 400
SAMPLES_ORPHAN = 100
SWEEP_SAMPLES = 50

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()


class PredicateFormatAnalyzer:
    """Impara il formato atteso da ogni predicato analizzando i dati di training."""

    def __init__(self, training_rows):
        self.stats = defaultdict(lambda: {
            "token_counts": [],
            "char_lengths": [],
            "has_digits": [],
            "uppercase_ratio": [],
            "samples": set()
        })
        self._analyze(training_rows)
        self._compute_stats()

    def _analyze(self, rows):
        for row in rows:
            # Estrai predicato e valore dal target "<PRED> value"
            match = re.match(r'(<[^>]+>)\s*(.+)', row['target'])
            if match:
                pred, val = match.groups()
                s = self.stats[pred]
                s["token_counts"].append(len(val.split()))
                s["char_lengths"].append(len(val))
                s["has_digits"].append(1.0 if re.search(r'\d', val) else 0.0)
                # Uppercase ratio (quante lettere sono maiuscole)
                letters = [c for c in val if c.isalpha()]
                if letters:
                    s["uppercase_ratio"].append(sum(1 for c in letters if c.isupper()) / len(letters))
                s["samples"].add(val.lower().strip())

    def _compute_stats(self):
        """Calcola statistiche aggregate per ogni predicato."""
        for pred, s in self.stats.items():
            if s["token_counts"]:
                s["expected_tokens"] = (
                    np.percentile(s["token_counts"], 5),
                    np.percentile(s["token_counts"], 95)
                )
                s["median_tokens"] = np.median(s["token_counts"])
            else:
                s["expected_tokens"] = (1, 10)
                s["median_tokens"] = 2

            if s["char_lengths"]:
                s["expected_length"] = (
                    np.percentile(s["char_lengths"], 5),
                    np.percentile(s["char_lengths"], 95)
                )
                s["median_length"] = np.median(s["char_lengths"])
            else:
                s["expected_length"] = (1, 100)
                s["median_length"] = 20

            s["digit_ratio"] = np.mean(s["has_digits"]) if s["has_digits"] else 0.5
            s["avg_uppercase"] = np.mean(s["uppercase_ratio"]) if s["uppercase_ratio"] else 0.5

        # Log statistiche
        print("\n    [FormatAnalyzer] Learned formats:")
        for pred, s in sorted(self.stats.items()):
            print(f"      {pred:20} | tokens: {s['expected_tokens'][0]:.0f}-{s['expected_tokens'][1]:.0f} "
                  f"| len: {s['expected_length'][0]:.0f}-{s['expected_length'][1]:.0f} "
                  f"| digits: {s['digit_ratio']:.0%} | samples: {len(s['samples'])}")

    def format_score(self, predicate: str, generated: str) -> float:
        """Score 0-1 basato su quanto il generato rispetta il formato appreso."""
        s = self.stats.get(predicate)
        if not s or not generated.strip():
            return 0.0

        score = 1.0
        tokens = len(generated.split())
        length = len(generated)
        has_digit = bool(re.search(r'\d', generated))

        # 1. Token count compliance (peso alto)
        min_t, max_t = s["expected_tokens"]
        if tokens < min_t * 0.5 or tokens > max_t * 2:
            score *= 0.3  # Fuori range drasticamente
        elif not (min_t <= tokens <= max_t):
            score *= 0.7  # Fuori range moderatamente

        # 2. Length compliance
        min_l, max_l = s["expected_length"]
        if length < min_l * 0.3 or length > max_l * 2:
            score *= 0.4  # Troppo corto o lungo
        elif not (min_l * 0.7 <= length <= max_l * 1.3):
            score *= 0.8

        # 3. Digit pattern compliance
        if s["digit_ratio"] > 0.8 and not has_digit:
            score *= 0.5  # Ci aspettiamo cifre ma non ce ne sono
        elif s["digit_ratio"] < 0.2 and has_digit:
            score *= 0.7  # Non ci aspettiamo cifre ma ce ne sono

        return score

    def is_novel(self, predicate: str, generated: str) -> bool:
        """Verifica se il valore generato è nuovo (non nel training set)."""
        s = self.stats.get(predicate)
        if not s:
            return True
        return generated.lower().strip() not in s["samples"]


def calculate_creative_score(orig: str, gen: str, predicate: str,
                             analyzer: PredicateFormatAnalyzer) -> dict:
    """
    Score multi-dimensionale per valutare creatività e format compliance.

    Returns dict con:
    - format_score: rispetto del formato (0-1)
    - novelty_score: è un valore nuovo? (0 o 1)
    - creativity_score: non è una copia? (0-1)
    - total_score: score combinato
    """
    gen_clean = gen.strip()
    orig_clean = orig.strip()

    if len(gen_clean) < 2:
        return {"format": 0, "novelty": 0, "creativity": 0, "total": 0}

    # 1. Format compliance (40% del peso)
    format_score = analyzer.format_score(predicate, gen_clean)

    # 2. Novelty - è un valore mai visto? (30% del peso)
    novelty_score = 1.0 if analyzer.is_novel(predicate, gen_clean) else 0.3

    # 3. Creativity - non è una copia esatta? (30% del peso)
    sim = SequenceMatcher(None, orig_clean.lower(), gen_clean.lower()).ratio()
    if sim > 0.95:
        creativity_score = 0.1  # Quasi identico = non creativo
    elif sim > 0.85:
        creativity_score = 0.5  # Molto simile
    elif sim > 0.5:
        creativity_score = 1.0  # Sweet spot: simile ma diverso
    elif sim > 0.3:
        creativity_score = 0.8  # Abbastanza diverso
    else:
        creativity_score = 0.4  # Troppo diverso, potrebbe essere garbage

    # Score totale pesato
    total = (format_score * 0.4) + (novelty_score * 0.3) + (creativity_score * 0.3)

    return {
        "format": format_score,
        "novelty": novelty_score,
        "creativity": creativity_score,
        "total": total
    }

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

    # Analizza formati dai dati di training
    format_analyzer = PredicateFormatAnalyzer(train_rows)

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
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            num_train_epochs=EPOCHS, 
            learning_rate=5e-5, 
            fp16=True, 
            report_to="none", 
            save_strategy="no",
            dataloader_num_workers=8,
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

    # 4. SWEEP CHIRURGICO PER BASE (con Temperature e Format-Aware Scoring)
    print(f"\n>>> PHASE 2: PARAMETER OPTIMIZATION (Format-Aware + Temperature)")
    alphas = [0.1, 0.2, 0.3]
    noises = [0.02, 0.05, 0.1]
    beams  = [1, 3, 5]
    temperatures = [0.7, 1.0, 1.3]

    results = []
    total_configs = len(alphas) * len(noises) * len(beams) * len(temperatures)
    config_idx = 0

    for a in alphas:
        for n in noises:
            for b in beams:
                for t in temperatures:
                    config_idx += 1
                    interpolator.latent_noise_std = n
                    interpolator.gen_num_beams = b
                    interpolator.gen_temperature = t
                    interpolator.gen_do_sample = True  # Abilita sampling per usare temperature

                    scores = {"format": [], "novelty": [], "creativity": [], "total": []}
                    for pred, v1, v2 in aligned_test:
                        res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                        s = calculate_creative_score(v1, res, pred, format_analyzer)
                        for k in scores:
                            scores[k].append(s[k])

                    avg_scores = {k: np.mean(v) if v else 0 for k, v in scores.items()}
                    results.append({
                        "a": a, "n": n, "b": b, "t": t,
                        "format": avg_scores["format"],
                        "novelty": avg_scores["novelty"],
                        "creativity": avg_scores["creativity"],
                        "total": avg_scores["total"]
                    })
                    print(f"    [{config_idx}/{total_configs}] A={a} N={n} B={b} T={t} -> "
                          f"F={avg_scores['format']:.2f} N={avg_scores['novelty']:.2f} "
                          f"C={avg_scores['creativity']:.2f} | Total={avg_scores['total']:.3f}")

    results.sort(key=lambda x: x['total'], reverse=True)
    best = results[0]
    print("\n    TOP 5 CONFIGURATIONS:")
    print(tabulate(results[:5], headers="keys", floatfmt=".3f"))
    print(f"\n    WINNING CONFIG: A={best['a']} N={best['n']} B={best['b']} T={best['t']}")

    # 5. REPORT FINALE CON SCORING DETTAGLIATO
    print("\n" + "="*100)
    print(f" ULTIMATE {MODEL_NAME.upper()} REPORT (Format-Aware) ".center(100))
    print("="*100)
    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_temperature = best['t']
    interpolator.gen_do_sample = True

    output_file = "massive_base_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA {MODEL_NAME} REPORT\n")
        f.write(f"Best Config: alpha={best['a']}, noise={best['n']}, beams={best['b']}, temp={best['t']}\n")
        f.write(f"Scores: Format={best['format']:.3f}, Novelty={best['novelty']:.3f}, "
                f"Creativity={best['creativity']:.3f}, Total={best['total']:.3f}\n\n")
        f.write("-"*120 + "\n")

        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower() != vt.lower() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        report_data, count, a_preds = [], 0, list(aligned_by_pred.keys())
        total_scores = {"format": [], "novelty": [], "creativity": [], "total": []}

        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aug, _ = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])

                    # Calcola score per questo sample
                    s = calculate_creative_score(v1, aug, p_tok, format_analyzer)
                    for k in total_scores:
                        total_scores[k].append(s[k])

                    f.write(f"{count+1:03d} | {p_tok:20} | {v1[:25]:25} | {v2[:25]:25} | {aug[:30]:30} | "
                            f"F={s['format']:.2f} N={s['novelty']:.2f} C={s['creativity']:.2f}\n")

                    if count < 20:
                        report_data.append([
                            count+1, p_tok[:15], v1[:18], v2[:18], aug[:25],
                            f"{s['format']:.1f}", f"{s['novelty']:.1f}", f"{s['creativity']:.1f}"
                        ])
                    count += 1
                else:
                    a_preds.remove(p_tok)
                if count >= SAMPLES_ALIGNED:
                    break

        # Summary finale
        f.write("-"*120 + "\n")
        f.write(f"AVERAGE SCORES: Format={np.mean(total_scores['format']):.3f}, "
                f"Novelty={np.mean(total_scores['novelty']):.3f}, "
                f"Creativity={np.mean(total_scores['creativity']):.3f}, "
                f"Total={np.mean(total_scores['total']):.3f}\n")

    print(f"\n>>> SUCCESS: Report saved to {output_file}")
    print(f"    Final Averages: F={np.mean(total_scores['format']):.3f} "
          f"N={np.mean(total_scores['novelty']):.3f} C={np.mean(total_scores['creativity']):.3f}")
    print("\n    SAMPLE OUTPUTS:")
    print(tabulate(report_data, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED", "F", "N", "C"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
