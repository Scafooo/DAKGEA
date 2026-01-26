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

# --- CONFIGURAZIONE RTX 4090 ---
MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 128
GRAD_ACCUMULATION = 4
EPOCHS = 15
SAMPLES_ALIGNED = 400
SWEEP_SAMPLES = 50

torch.backends.cudnn.benchmark = True
logger = get_logger("MassiveSweep")

def clean_val(text):
    return re.sub(r'<[^>]+>\s*', '', text).strip()

class PredicateFormatAnalyzer:
    """Analizza le statistiche di formato per ogni predicato dal training data."""

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
                s["median_tokens"] = np.median(s["token_counts"])

    def format_score(self, pred, gen):
        """Calcola score di compliance al formato atteso (0-1)."""
        s = self.stats.get(pred)
        if not s or not gen.strip():
            return 0.0
        score, tokens, length = 1.0, len(gen.split()), len(gen)
        if not (s["expected_tokens"][0] <= tokens <= s["expected_tokens"][1]):
            score *= 0.7
        if not (s["expected_length"][0] <= length <= s["expected_length"][1]):
            score *= 0.7
        return score

    def is_short_value_predicate(self, pred):
        """Determina se il predicato ha tipicamente valori corti (es. nomi)."""
        s = self.stats.get(pred)
        if not s:
            return False
        return s.get("median_tokens", 10) <= 4


def calculate_diversity_score(orig, gen, pred, analyzer, other=None):
    """
    Nuova metrica di scoring che premia QUALSIASI diversità.

    Filosofia: per i nomi brevi, anche "John Smith" -> "J. Smith" è buono.
    Non pretendiamo semantic leap, basta che sia DIVERSO e ben formato.

    Returns:
        dict con:
        - total: score finale (0-1)
        - diversity: quanto è diverso dall'originale (0-1)
        - format: compliance al formato (0-1)
        - transform: tipo di trasformazione
    """
    gen_c = gen.strip()
    orig_c = orig.strip()
    other_c = other.strip() if other else ""

    # Garbage detection
    if len(gen_c) < 2:
        return {"total": 0, "diversity": 0, "format": 0, "transform": "garbage_empty"}

    # 1. FORMAT COMPLIANCE (0-1)
    f_score = analyzer.format_score(pred, gen_c)

    # 2. DIVERSITY SCORE basato su edit distance (0-1)
    # Usiamo 1 - similarity come misura di diversità
    sim_orig = SequenceMatcher(None, orig_c.lower(), gen_c.lower()).ratio()
    sim_other = SequenceMatcher(None, other_c.lower(), gen_c.lower()).ratio() if other_c else 1.0

    # La diversità è quanto siamo lontani da ENTRAMBI gli input
    diversity_from_orig = 1.0 - sim_orig
    diversity_from_other = 1.0 - sim_other

    # Prendiamo la diversità minima (deve essere diverso da entrambi)
    min_diversity = min(diversity_from_orig, diversity_from_other)

    # Classifica il tipo di trasformazione
    if sim_orig > 0.98:
        ttype = "identity"
        d_score = 0.0  # Copia esatta = 0 punti
    elif sim_orig > 0.90:
        ttype = "near_copy"
        d_score = 0.2  # Quasi copia
    elif sim_orig > 0.70:
        ttype = "minor_variation"
        d_score = 0.5  # Variazione minore ma OK
    elif sim_orig > 0.50:
        ttype = "good_variation"
        d_score = 0.8  # Buona variazione
    elif sim_orig > 0.30:
        ttype = "strong_transform"
        d_score = 1.0  # Trasformazione significativa
    else:
        # Troppo diverso potrebbe essere garbage
        # Verifichiamo che abbia almeno qualche parola in comune
        gen_words = set(re.findall(r'\w+', gen_c.lower()))
        orig_words = set(re.findall(r'\w+', orig_c.lower()))
        other_words = set(re.findall(r'\w+', other_c.lower()))
        source_words = orig_words | other_words

        common = gen_words & source_words
        if len(common) > 0 or f_score > 0.5:
            ttype = "creative_transform"
            d_score = 0.9  # Creativo ma mantiene qualche connessione
        else:
            ttype = "garbage_unrelated"
            d_score = 0.1  # Probabilmente garbage

    # Bonus per parole nuove (semantic novelty)
    gen_words = set(re.findall(r'\w+', gen_c.lower()))
    orig_words = set(re.findall(r'\w+', orig_c.lower()))
    other_words = set(re.findall(r'\w+', other_c.lower()))
    source_words = orig_words | other_words
    new_words = [w for w in (gen_words - source_words) if len(w) > 2]

    novelty_bonus = min(0.2, len(new_words) * 0.05)  # Max +0.2 bonus

    # Score finale: Format (30%) + Diversity (70%) + bonus
    total = (f_score * 0.3) + (d_score * 0.7) + novelty_bonus
    total = min(1.0, total)  # Cap a 1.0

    return {
        "total": total,
        "diversity": d_score,
        "format": f_score,
        "transform": ttype,
        "sim_orig": sim_orig,
        "new_words": len(new_words)
    }


# Alias per backward compatibility
def calculate_creative_score(orig, gen, pred, analyzer, other=None):
    """Alias per la nuova funzione di scoring."""
    return calculate_diversity_score(orig, gen, pred, analyzer, other)

def run_massive_sweep():
    print("\n" + "█"*100); print(f"█ RTX 4090: STABLE EVALUATION (POST-FEEDBACK) ".center(98) + "█"); print("█"*100)

    reader = OpeneaDatasetReader()
    dataset = reader.read(str(PROJECT_ROOT / "data/raw/openea/BBC_DB"))
    builder = MixupDataBuilder()
    train_rows, canonical_map = builder.build_training_data(dataset)
    format_analyzer = PredicateFormatAnalyzer(train_rows)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_base_v1"

    interpolator = MixupBartInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)
    interpolator.set_predicate_mapping(canonical_map)

    if not (Path(out_dir) / "pytorch_model.bin").exists() and not (Path(out_dir) / "model.safetensors").exists():
        print("    Starting Training..."); hf_ds = HFDataset.from_list(train_rows).map(lambda b: interpolator.tokenizer(b["input"], text_target=b["target"], max_length=64, truncation=True, padding="max_length"), batched=True)
        args = Seq2SeqTrainingArguments(output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUMULATION, num_train_epochs=EPOCHS, learning_rate=5e-5, fp16=True, report_to="none", save_strategy="no")
        trainer = Seq2SeqTrainer(model=interpolator.model, args=args, train_dataset=hf_ds, data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model))
        trainer.train(); interpolator.model.save_pretrained(out_dir); interpolator.tokenizer.save_pretrained(out_dir)

    # Preparazione Subset Clean
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

    # GRID SEARCH con range ESTREMI per forzare creatività
    print(f"\n>>> PHASE 2: PARAMETER OPTIMIZATION (AGGRESSIVE RANGES)")
    print("    Testing extreme temperature/noise to force diversity...")

    # NUOVI RANGE PIÙ AGGRESSIVI
    alphas = [0.3, 0.5, 0.7]              # Più mixing (era 0.1, 0.3, 0.5)
    noises = [0.05, 0.1, 0.15, 0.2]       # Più perturbazione (era 0.02, 0.05, 0.1)
    beams = [1, 3]                         # Beam search (1=sampling puro, 3=un po' di beam)
    temps = [1.0, 1.5, 2.0, 2.5]          # Molto più alte (era 0.7, 1.0, 1.3)

    results = []
    total_configs = len(alphas) * len(noises) * len(beams) * len(temps)
    config_idx = 0

    for a in alphas:
        for n in noises:
            for b in beams:
                for t in temps:
                    config_idx += 1
                    interpolator.latent_noise_std = n
                    interpolator.gen_num_beams = b
                    interpolator.gen_temperature = t

                    scores_detail = []
                    transform_counts = defaultdict(int)

                    for pred, v1, v2 in aligned_test:
                        a1, a2 = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)

                        sc1 = calculate_diversity_score(v1, a1, pred, format_analyzer, v2)
                        sc2 = calculate_diversity_score(v2, a2, pred, format_analyzer, v1)

                        scores_detail.append((sc1["total"] + sc2["total"]) / 2)
                        transform_counts[sc1["transform"]] += 1
                        transform_counts[sc2["transform"]] += 1

                    avg_score = np.mean(scores_detail)
                    identity_pct = transform_counts.get("identity", 0) / (len(aligned_test) * 2) * 100

                    results.append({
                        "a": a, "n": n, "b": b, "t": t,
                        "score": avg_score,
                        "identity_pct": identity_pct,
                        "transforms": dict(transform_counts)
                    })

                    print(f"    [{config_idx:3d}/{total_configs}] A={a:.1f} N={n:.2f} B={b} T={t:.1f} -> "
                          f"Score: {avg_score:.3f} | Identity: {identity_pct:.0f}%")

    # Sort by score (higher is better) and show top configs
    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    print("\n" + "="*100)
    print(" TOP 5 CONFIGURATIONS ".center(100))
    print("="*100)
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}: A={r['a']:.1f} N={r['n']:.2f} B={r['b']} T={r['t']:.1f} -> "
              f"Score={r['score']:.3f} | Identity={r['identity_pct']:.0f}%")

    # REPORT FINALE MASSIVO
    print("\n" + "="*100)
    print(" GENERATING FINAL REPORT WITH BEST CONFIG ".center(100))
    print("="*100)

    interpolator.latent_noise_std = best['n']
    interpolator.gen_num_beams = best['b']
    interpolator.gen_temperature = best['t']

    output_file = "massive_diversity_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write(" DAKGEA DIVERSITY REPORT - AGGRESSIVE PARAMETER SWEEP \n")
        f.write("=" * 110 + "\n\n")
        f.write(f"Best Config: alpha={best['a']}, noise={best['n']}, beams={best['b']}, temp={best['t']}\n")
        f.write(f"Score: {best['score']:.3f} | Identity Rate: {best['identity_pct']:.1f}%\n")
        f.write(f"Transform Distribution: {best['transforms']}\n\n")
        f.write("-" * 110 + "\n\n")

        aligned_by_pred = defaultdict(list)
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower().strip() != vt.lower().strip() and len(vs) > 3 and canonical_map.get(ps) == canonical_map.get(pt):
                        aligned_by_pred[canonical_map[ps]].append((canonical_map[ps], vs, vt))

        count = 0
        pred_stats = defaultdict(lambda: {"total": 0, "identity": 0, "scores": []})
        a_preds = list(aligned_by_pred.keys())

        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])

                    # Calcola score per questa coppia
                    sc1 = calculate_diversity_score(v1, aa, p_tok, format_analyzer, v2)
                    sc2 = calculate_diversity_score(v2, ab, p_tok, format_analyzer, v1)
                    avg_sc = (sc1["total"] + sc2["total"]) / 2

                    # Aggiorna statistiche per predicato
                    pred_stats[p_tok]["total"] += 2
                    pred_stats[p_tok]["scores"].append(avg_sc)
                    if sc1["transform"] == "identity":
                        pred_stats[p_tok]["identity"] += 1
                    if sc2["transform"] == "identity":
                        pred_stats[p_tok]["identity"] += 1

                    # Scrivi nel report
                    f.write(f"{count+1:03d} | {p_tok:15} | VAL A: {v1[:25]:25} -> AUG A': {aa[:30]:30}\n")
                    f.write(f"    | {' ':15} | VAL B: {v2[:25]:25} -> AUG B': {ab[:30]:30}\n")
                    f.write(f"    | {' ':15} | Score: {avg_sc:.2f} | Type: {sc1['transform']}/{sc2['transform']} | Voto:[ ]/5\n")
                    f.write(f"    | {' ':15} | Note: [__________________________________________________]\n")
                    f.write("-" * 110 + "\n")
                    count += 1
                else:
                    a_preds.remove(p_tok)
                if count >= SAMPLES_ALIGNED:
                    break

        # Scrivi riepilogo per predicato
        f.write("\n" + "=" * 110 + "\n")
        f.write(" RIEPILOGO PER PREDICATO \n")
        f.write("=" * 110 + "\n\n")
        for p_tok, stats in sorted(pred_stats.items()):
            identity_rate = stats["identity"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_score = np.mean(stats["scores"]) if stats["scores"] else 0
            f.write(f"{p_tok:20} | Samples: {stats['total']:3d} | Avg Score: {avg_score:.2f} | Identity: {identity_rate:5.1f}%\n")

    print(f">>> SUCCESS: Report saved to {output_file}")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
