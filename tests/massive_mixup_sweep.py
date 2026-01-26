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


def classify_transformation(orig: str, other: str, gen: str) -> str:
    """
    Classifica il tipo di trasformazione effettuata.

    Returns: 'identity', 'token_swap', 'partial_merge', 'creative', 'garbage'
    """
    orig_l, other_l, gen_l = orig.lower().strip(), other.lower().strip(), gen.lower().strip()
    orig_tokens = set(orig_l.split())
    other_tokens = set(other_l.split())
    gen_tokens = set(gen_l.split())

    # Similarity scores
    sim_orig = SequenceMatcher(None, orig_l, gen_l).ratio()
    sim_other = SequenceMatcher(None, other_l, gen_l).ratio()

    # Identity: quasi identico all'originale
    if sim_orig > 0.95:
        return "identity"

    # Identity con l'altro valore
    if sim_other > 0.95:
        return "identity_other"

    # Token swap: contiene token da entrambi
    orig_in_gen = len(orig_tokens & gen_tokens)
    other_in_gen = len(other_tokens & gen_tokens)
    if orig_in_gen > 0 and other_in_gen > 0:
        return "token_swap"

    # Partial merge: molto simile ma con modifiche
    if 0.6 <= sim_orig <= 0.95 or 0.6 <= sim_other <= 0.95:
        return "partial_merge"

    # Creative: abbastanza diverso ma sensato (lunghezza ragionevole)
    if 0.3 <= sim_orig <= 0.6 and len(gen_l) >= 3:
        return "creative"

    # Garbage: troppo diverso o troppo corto
    return "garbage"


def calculate_creative_score(orig: str, gen: str, predicate: str,
                             analyzer: PredicateFormatAnalyzer,
                             other: str = None) -> dict:
    """
    Score multi-dimensionale per valutare creatività e format compliance.

    Returns dict con:
    - format_score: rispetto del formato (0-1)
    - novelty_score: è un valore nuovo? (0 o 1)
    - creativity_score: non è una copia? (0-1)
    - total_score: score combinato
    - transform_type: tipo di trasformazione
    """
    gen_clean = gen.strip()
    orig_clean = orig.strip()
    other_clean = other.strip() if other else ""

    if len(gen_clean) < 2:
        return {"format": 0, "novelty": 0, "creativity": 0, "total": 0, "transform": "garbage"}

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

    # 4. Classifica trasformazione
    transform_type = classify_transformation(orig_clean, other_clean, gen_clean)

    # Score totale pesato
    total = (format_score * 0.4) + (novelty_score * 0.3) + (creativity_score * 0.3)

    return {
        "format": format_score,
        "novelty": novelty_score,
        "creativity": creativity_score,
        "total": total,
        "transform": transform_type
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

    # 3. PREPARAZIONE TEST SUBSET (SOLO DATI REALI PULITI)
    print("    Extracting clean evaluation pairs directly from KGs...")
    aligned_test = []
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target
    
    # Iteriamo sulle entità allineate ufficiali del dataset
    for s_uri, t_uri in list(dataset.aligned_entities):
        # Estraiamo tutti i letterali puliti per questa coppia di entità
        s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
        t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
        
        for ps, vs in s_lits.items():
            for pt, vt in t_lits.items():
                # Se i predicati sono semanticamente accoppiati
                if canonical_map.get(ps) == canonical_map.get(pt):
                    # Se i valori sono diversi (per testare la traduzione/mixup)
                    if vs.lower().strip() != vt.lower().strip() and len(vs) > 3:
                        aligned_test.append((canonical_map[ps], vs, vt))
        
        if len(aligned_test) >= SWEEP_SAMPLES: break
    
    print(f"    Clean Test Set size: {len(aligned_test)}")

    # 4. SWEEP CHIRURGICO PER BASE (con Temperature e Format-Aware Scoring)
    print(f"\n>>> PHASE 2: PARAMETER OPTIMIZATION (Format-Aware + Temperature)")
    print(f"    Testing {SWEEP_SAMPLES} samples per configuration\n")
    alphas = [0.1, 0.2, 0.3]
    noises = [0.02, 0.05, 0.1]
    beams  = [1, 3, 5]
    temperatures = [0.7, 1.0, 1.3]

    results = []
    total_configs = len(alphas) * len(noises) * len(beams) * len(temperatures)
    config_idx = 0
    NUM_EXAMPLES = 3  # Esempi da mostrare per configurazione

    for a in alphas:
        for n in noises:
            for b in beams:
                for t in temperatures:
                    config_idx += 1
                    interpolator.latent_noise_std = n
                    interpolator.gen_num_beams = b
                    interpolator.gen_temperature = t
                    interpolator.gen_do_sample = True

                    scores = {"format": [], "novelty": [], "creativity": [], "total": []}
                    transforms = defaultdict(int)
                    examples = []

                    for pred, v1, v2 in aligned_test:
                        res, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=a)
                        s = calculate_creative_score(v1, res, pred, format_analyzer, other=v2)
                        for k in ["format", "novelty", "creativity", "total"]:
                            scores[k].append(s[k])
                        transforms[s["transform"]] += 1

                        # Salva primi esempi
                        if len(examples) < NUM_EXAMPLES:
                            examples.append({
                                "pred": pred, "v1": v1, "v2": v2,
                                "gen": res, "transform": s["transform"],
                                "score": s["total"]
                            })

                    avg_scores = {k: np.mean(v) if v else 0 for k, v in scores.items()}
                    total_samples = sum(transforms.values())
                    transform_pcts = {k: v/total_samples*100 for k, v in transforms.items()}

                    results.append({
                        "a": a, "n": n, "b": b, "t": t,
                        "format": avg_scores["format"],
                        "novelty": avg_scores["novelty"],
                        "creativity": avg_scores["creativity"],
                        "total": avg_scores["total"],
                        "transforms": dict(transforms),
                        "examples": examples
                    })

                    # Output dettagliato
                    print(f"{'─'*90}")
                    print(f"[{config_idx}/{total_configs}] Alpha={a} Noise={n} Beams={b} Temp={t}")
                    print(f"    Scores: Format={avg_scores['format']:.2f} Novelty={avg_scores['novelty']:.2f} "
                          f"Creativity={avg_scores['creativity']:.2f} → Total={avg_scores['total']:.3f}")

                    # Transform breakdown
                    transform_str = " | ".join([f"{k}:{v:.0f}%" for k, v in sorted(transform_pcts.items())])
                    print(f"    Transforms: {transform_str}")

                    # Esempi
                    print(f"    Examples:")
                    for ex in examples:
                        icon = "✓" if ex["transform"] in ["token_swap", "partial_merge", "creative"] else "✗"
                        print(f"      {icon} {ex['pred'][:12]:12} \"{ex['v1'][:15]}\" + \"{ex['v2'][:15]}\" → \"{ex['gen'][:20]}\" [{ex['transform']}]")

    results.sort(key=lambda x: x['total'], reverse=True)
    best = results[0]

    print(f"\n{'═'*90}")
    print(" TOP 5 CONFIGURATIONS ".center(90, "═"))
    print(f"{'═'*90}")
    top5_display = [{k: v for k, v in r.items() if k not in ["transforms", "examples"]} for r in results[:5]]
    print(tabulate(top5_display, headers="keys", floatfmt=".3f"))

    print(f"\n{'═'*90}")
    print(f" WINNING CONFIG: Alpha={best['a']} Noise={best['n']} Beams={best['b']} Temp={best['t']} ".center(90, "═"))
    print(f"{'═'*90}")

    # Mostra transform distribution della configurazione vincente
    best_transforms = best.get("transforms", {})
    total_t = sum(best_transforms.values()) or 1
    print("\n    Transform Distribution (Best Config):")
    for ttype in ["identity", "identity_other", "token_swap", "partial_merge", "creative", "garbage"]:
        count = best_transforms.get(ttype, 0)
        pct = count / total_t * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        icon = "✓" if ttype in ["token_swap", "partial_merge", "creative"] else "✗"
        print(f"      {icon} {ttype:15} {bar} {pct:5.1f}% ({count})")

    # Mostra esempi della configurazione vincente
    print("\n    Best Config Examples:")
    for ex in best.get("examples", []):
        icon = "✓" if ex["transform"] in ["token_swap", "partial_merge", "creative"] else "✗"
        print(f"      {icon} {ex['pred'][:15]:15} \"{ex['v1'][:20]}\" + \"{ex['v2'][:20]}\"")
        print(f"        → \"{ex['gen']}\" [{ex['transform']}, score={ex['score']:.2f}]")

    # 5. REPORT FINALE CON SCORING DETTAGLIATO E ANALISI PER PREDICATO
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
        f.write("-"*130 + "\n")

        aligned_by_pred = defaultdict(list)
        # Estraiamo TUTTE le coppie clean dal dataset originale
        for s_uri, t_uri in dataset.aligned_entities:
            s_lits = {str(p): str(o) for _, p, o in kg_src.triples((s_uri, None, None)) if isinstance(o, Literal)}
            t_lits = {str(p): str(o) for _, p, o in kg_tgt.triples((t_uri, None, None)) if isinstance(o, Literal)}
            for ps, vs in s_lits.items():
                for pt, vt in t_lits.items():
                    if vs.lower().strip() != vt.lower().strip() and len(vs) > 3:
                        if canonical_map.get(ps) == canonical_map.get(pt):
                            p_tok = canonical_map[ps]
                            aligned_by_pred[p_tok].append((p_tok, vs, vt))

        report_data, count, a_preds = [], 0, list(aligned_by_pred.keys())
        total_scores = {"format": [], "novelty": [], "creativity": [], "total": []}
        total_transforms = defaultdict(int)

        # Per-predicate tracking
        pred_scores = defaultdict(lambda: {"format": [], "novelty": [], "creativity": [], "total": [], "transforms": defaultdict(int), "count": 0})

        while count < SAMPLES_ALIGNED and a_preds:
            for p_tok in a_preds[:]:
                if aligned_by_pred[p_tok]:
                    p_uri, v1, v2 = aligned_by_pred[p_tok].pop(random.randrange(len(aligned_by_pred[p_tok])))
                    aug_a, aug_b = interpolator.interpolate_pair(v1, v2, predicate=p_tok, alpha=best['a'])

                    # Calcola score per entrambi
                    s_a = calculate_creative_score(v1, aug_a, p_tok, format_analyzer, other=v2)
                    s_b = calculate_creative_score(v2, aug_b, p_tok, format_analyzer, other=v1)
                    
                    # Media per le statistiche
                    for k in ["format", "novelty", "creativity", "total"]:
                        val = (s_a[k] + s_b[k]) / 2
                        total_scores[k].append(val)
                        pred_scores[p_tok][k].append(val)

                    total_transforms[s_a["transform"]] += 1
                    total_transforms[s_b["transform"]] += 1
                    pred_scores[p_tok]["count"] += 1

                    # Scrittura su file con AUG A' e AUG B'
                    f.write(f"{count+1:03d} | {p_tok:15} | VAL A: {v1[:20]:20} -> AUG A': {aug_a[:25]:25} | Q:{s_a['total']*100:2.0f}% | {s_a['transform']:10}\n")
                    f.write(f"    | {' ':15} | VAL B: {v2[:20]:20} -> AUG B': {aug_b[:25]:25} | Q:{s_b['total']*100:2.0f}% | {s_b['transform']:10} | Voto:[ ]/5\n")
                    f.write(f"    | {' ':15} | Note: [__________________________________________________]\n")
                    f.write("-" * 130 + "\n")

                    if count < 30:
                        report_data.append([
                            count+1, p_tok[:10], v1[:12], aug_a[:15], v2[:12], aug_b[:15],
                            f"{(s_a['total']+s_b['total'])*50:.0f}%", "[ ]/5"
                        ])
                    count += 1
                else:
                    a_preds.remove(p_tok)
                if count >= SAMPLES_ALIGNED:
                    break

        # Summary finale nel file
        f.write("\n" + "="*130 + "\n")
        f.write("SUMMARY\n")
        f.write("="*130 + "\n\n")

        f.write(f"AVERAGE SCORES: Format={np.mean(total_scores['format']):.3f}, "
                f"Novelty={np.mean(total_scores['novelty']):.3f}, "
                f"Creativity={np.mean(total_scores['creativity']):.3f}, "
                f"Total={np.mean(total_scores['total']):.3f}\n\n")

        # Transform distribution nel file
        f.write("TRANSFORM DISTRIBUTION:\n")
        total_t = sum(total_transforms.values()) or 1
        for ttype in ["identity", "identity_other", "token_swap", "partial_merge", "creative", "garbage"]:
            cnt = total_transforms.get(ttype, 0)
            pct = cnt / total_t * 100
            bar = "█" * int(pct / 2)
            f.write(f"  {ttype:15} {bar:50} {pct:5.1f}% ({cnt})\n")

        # Per-predicate analysis nel file
        f.write("\nPER-PREDICATE ANALYSIS:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Predicate':20} {'Format':>8} {'Novelty':>8} {'Creativity':>8} {'Total':>8} {'Samples':>8} {'Best Transform':>20}\n")
        f.write("-"*100 + "\n")

        for pred in sorted(pred_scores.keys()):
            ps = pred_scores[pred]
            if ps["count"] > 0:
                avg_f = np.mean(ps["format"])
                avg_n = np.mean(ps["novelty"])
                avg_c = np.mean(ps["creativity"])
                avg_t = np.mean(ps["total"])
                best_transform = max(ps["transforms"].items(), key=lambda x: x[1])[0] if ps["transforms"] else "N/A"
                f.write(f"{pred:20} {avg_f:8.3f} {avg_n:8.3f} {avg_c:8.3f} {avg_t:8.3f} {ps['count']:8d} {best_transform:>20}\n")

    # Console output
    print(f"\n>>> SUCCESS: Report saved to {output_file}")
    print(f"\n{'─'*90}")
    print(" FINAL AVERAGES ".center(90, "─"))
    print(f"{'─'*90}")
    print(f"    Format:     {np.mean(total_scores['format']):.3f}")
    print(f"    Novelty:    {np.mean(total_scores['novelty']):.3f}")
    print(f"    Creativity: {np.mean(total_scores['creativity']):.3f}")
    print(f"    TOTAL:      {np.mean(total_scores['total']):.3f}")

    # Transform distribution console
    print(f"\n{'─'*90}")
    print(" TRANSFORM DISTRIBUTION ".center(90, "─"))
    print(f"{'─'*90}")
    total_t = sum(total_transforms.values()) or 1
    good_transforms = total_transforms.get("token_swap", 0) + total_transforms.get("partial_merge", 0) + total_transforms.get("creative", 0)
    bad_transforms = total_transforms.get("identity", 0) + total_transforms.get("identity_other", 0) + total_transforms.get("garbage", 0)

    for ttype in ["identity", "identity_other", "token_swap", "partial_merge", "creative", "garbage"]:
        cnt = total_transforms.get(ttype, 0)
        pct = cnt / total_t * 100
        bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
        icon = "✓" if ttype in ["token_swap", "partial_merge", "creative"] else "✗"
        print(f"    {icon} {ttype:15} {bar} {pct:5.1f}% ({cnt})")

    print(f"\n    Good transforms: {good_transforms/total_t*100:.1f}% | Bad transforms: {bad_transforms/total_t*100:.1f}%")

    # Per-predicate summary console
    print(f"\n{'─'*90}")
    print(" PER-PREDICATE ANALYSIS ".center(90, "─"))
    print(f"{'─'*90}")
    pred_summary = []
    for pred in sorted(pred_scores.keys()):
        ps = pred_scores[pred]
        if ps["count"] > 0:
            good = ps["transforms"].get("token_swap", 0) + ps["transforms"].get("partial_merge", 0) + ps["transforms"].get("creative", 0)
            bad = ps["transforms"].get("identity", 0) + ps["transforms"].get("identity_other", 0) + ps["transforms"].get("garbage", 0)
            pred_summary.append([
                pred[:15],
                f"{np.mean(ps['format']):.2f}",
                f"{np.mean(ps['novelty']):.2f}",
                f"{np.mean(ps['creativity']):.2f}",
                f"{np.mean(ps['total']):.2f}",
                ps["count"],
                f"{good}/{bad}"
            ])
    print(tabulate(pred_summary, headers=["Predicate", "Format", "Novelty", "Creat.", "Total", "N", "Good/Bad"], tablefmt="simple"))

    # Sample outputs
    print(f"\n{'─'*90}")
    print(" SAMPLE OUTPUTS ".center(90, "─"))
    print(f"{'─'*90}")
    print(tabulate(report_data[:20], headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED", "F", "N", "C", "Type"], tablefmt="grid"))

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run_massive_sweep()
