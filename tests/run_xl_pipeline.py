import sys
import torch
import logging
import random
import time
import numpy as np
import re
import os
from pathlib import Path
from collections import defaultdict
from rdflib import Literal
from sentence_transformers import SentenceTransformer, util

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder
from src.augmentation.methods.plm.scoring import calculate_pair_score, calculate_score
from src.augmentation.methods.plm.creative_variation_generator import CreativeVariationGenerator

# --- CONFIGURAZIONE CREATIVE VARIATION ---
MODEL_NAME = "google/flan-t5-xl"
BATCH_SIZE = 8
EPOCHS = 3
TOTAL_REPORT_SAMPLES = 200
MAX_ORPHANS_PER_PRED = 200
SWEEP_SAMPLES = 50

logger = get_logger("FlanT5XL_Creative")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
creative_gen = CreativeVariationGenerator()

# --- FILTRI QUALITÀ TRAINING DATA ---
UNICODE_GARBAGE_PATTERN = re.compile(r'u00[a-f0-9]{2}', re.IGNORECASE)
FILLER_WORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for'}

def has_unicode_garbage(text: str) -> bool:
    """Rileva caratteri unicode corrotti come u00e9, u00f3, etc."""
    return bool(UNICODE_GARBAGE_PATTERN.search(text))

def is_only_filler_difference(input_text: str, target_text: str) -> bool:
    """
    Rileva se l'unica differenza tra input e target è l'aggiunta/rimozione di filler words.

    "kinks" → "kinks the" → True (solo aggiunto "the")
    "the band" → "band" → True (solo rimosso "the")
    "john smith" → "john the smith" → True (solo aggiunto "the")
    "john smith" → "johnny smith" → False (variazione reale)
    """
    # Estrai il valore dopo ": " nel prompt
    if ": " in input_text:
        input_val = input_text.split(": ", 1)[1].strip().lower()
    else:
        input_val = input_text.strip().lower()

    target_val = target_text.strip().lower()

    # Se uguali, non è un problema di filler
    if input_val == target_val:
        return False

    # Tokenizza
    input_tokens = input_val.split()
    target_tokens = target_val.split()

    # Rimuovi filler da entrambi
    input_no_filler = [t for t in input_tokens if t not in FILLER_WORDS]
    target_no_filler = [t for t in target_tokens if t not in FILLER_WORDS]

    # Se senza filler sono uguali → l'unica differenza erano i filler
    if input_no_filler == target_no_filler:
        return True

    return False

def is_token_swap(input_text: str, target_text: str) -> bool:
    """Rileva se target è solo uno swap di token dell'input (stesso set di parole)."""
    # Estrai il valore dopo ": " nel prompt
    if ": " in input_text:
        input_val = input_text.split(": ", 1)[1].strip()
    else:
        input_val = input_text.strip()

    target_val = target_text.strip()

    # Normalizza e tokenizza
    input_tokens = set(input_val.lower().split())
    target_tokens = set(target_val.lower().split())

    # Se hanno esattamente gli stessi token ma in ordine diverso → swap
    if input_tokens == target_tokens and input_val.lower() != target_val.lower():
        return True
    return False

def min_edit_distance(s1: str, s2: str) -> int:
    """Calcola la distanza di Levenshtein tra due stringhe."""
    if len(s1) < len(s2):
        return min_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def extract_value_from_prompt(prompt: str) -> str:
    """Estrae il valore dopo ': ' dal prompt."""
    if ": " in prompt:
        return prompt.split(": ", 1)[1].strip()
    return prompt.strip()

def filter_training_data(rows: list, min_edit_ratio: float = 0.1) -> list:
    """Filtra righe con unicode garbage, token swap, filler difference, o troppo simili."""
    filtered = []
    unicode_removed = 0
    swap_removed = 0
    filler_removed = 0
    too_similar_removed = 0

    for row in rows:
        inp, tgt = row['input'], row['target']

        # Estrai valore dall'input per confronto
        inp_val = extract_value_from_prompt(inp)

        # Fix 1: Rimuovi unicode garbage
        if has_unicode_garbage(inp) or has_unicode_garbage(tgt):
            unicode_removed += 1
            continue

        # Fix 2: Rimuovi token swap puri
        if is_token_swap(inp, tgt):
            swap_removed += 1
            continue

        # Fix 3: Rimuovi coppie dove l'unica differenza è "the"/"and"/etc.
        if is_only_filler_difference(inp, tgt):
            filler_removed += 1
            continue

        # Fix 4: Rimuovi coppie troppo simili (edit distance < 10% della lunghezza)
        max_len = max(len(inp_val), len(tgt))
        if max_len > 0:
            edit_dist = min_edit_distance(inp_val.lower(), tgt.lower())
            edit_ratio = edit_dist / max_len
            if edit_ratio < min_edit_ratio and inp_val.lower() != tgt.lower():
                too_similar_removed += 1
                continue

        filtered.append(row)

    print(f"    [FILTER] Removed: {unicode_removed} unicode, {swap_removed} swaps, {filler_removed} filler, {too_similar_removed} too-similar")
    print(f"    [FILTER] Kept: {len(filtered)}/{len(rows)} ({100*len(filtered)/len(rows):.1f}%)")
    return filtered

def load_attr_names(dataset_path):
    attr_map = {}
    for i in [1, 2]:
        path = Path(dataset_path) / f"attribute_data/attr_names{i}"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2: attr_map[parts[0].strip()] = parts[1].strip()
    return attr_map

def clean_p(uri, attr_map):
    uri_str = str(uri)
    if uri_str in attr_map: return attr_map[uri_str].replace(' ', '_').lower()
    return uri_str.split('/')[-1].split('#')[-1].replace('>', '').replace('<', '').lower()

def run_xl_pipeline():
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 XL (3B) - CREATIVE VARIATION PIPELINE ".center(98) + "█"); print("█"*100)

    # 1. DATA
    dataset_path = str(PROJECT_ROOT / "data/raw/openea/BBC_DB")
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)
    builder = MixupDataBuilder()
    _, canonical_map = builder.build_training_data(dataset, max_pairs_per_pred=10)
    kg_src, kg_tgt = dataset.knowledge_graph_source, dataset.knowledge_graph_target

    print("    [1/4] Building Training Data (CREATIVE VARIATION Focus)...")
    t5_rows = []
    src_lits = defaultdict(list)
    for s, p, o in kg_src.triples((None, None, None)):
        if isinstance(o, Literal): src_lits[s].append((p, str(o)))
    tgt_lits = defaultdict(list)
    for s, p, o in kg_tgt.triples((None, None, None)):
        if isinstance(o, Literal): tgt_lits[s].append((p, str(o)))

    # A. ALIGNED - Le coppie allineate sono GIÀ variazioni naturali!
    # Input: valore originale → Target: variazione reale cross-KG
    # PRIORITÀ: coppie reali valgono 3x (duplicate)
    aligned_test_pool = defaultdict(list)
    real_pairs_count = 0
    synthetic_pairs_count = 0

    for s_uri, t_uri in dataset.aligned_entities:
        s_attrs = src_lits.get(s_uri, [])
        t_attrs = tgt_lits.get(t_uri, [])
        for ps, vs in s_attrs:
            p_name = clean_p(ps, attr_map).replace('_', ' ')
            for pt, vt in t_attrs:
                if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                    v1_c, v2_c = vs.strip().lower(), vt.strip().lower()
                    if v1_c != v2_c:
                        # BIDIREZIONALE: coppie REALI duplicate 3x per più peso
                        for _ in range(3):
                            t5_rows.append({"input": f"generate variation <{p_name}>: {vs}", "target": vt})
                            t5_rows.append({"input": f"generate variation <{p_name}>: {vt}", "target": vs})
                        real_pairs_count += 2

                        # EXTRA: Variazioni sintetiche (solo 30% probabilità, ridotto)
                        if random.random() < 0.3:
                            var_vs = creative_gen.generate(vs, vt, predicate=p_name)
                            if var_vs != vs and var_vs != vt:
                                t5_rows.append({"input": f"generate variation <{p_name}>: {vs}", "target": var_vs})
                                synthetic_pairs_count += 1
                        if random.random() < 0.3:
                            var_vt = creative_gen.generate(vt, vs, predicate=p_name)
                            if var_vt != vt and var_vt != vs:
                                t5_rows.append({"input": f"generate variation <{p_name}>: {vt}", "target": var_vt})
                                synthetic_pairs_count += 1

                    aligned_test_pool[p_name].append((vs, vt))
                    break

    print(f"    [ALIGNED] Real pairs: {real_pairs_count} (x3 weight), Synthetic: {synthetic_pairs_count}")

    # B. ORPHANS - NUOVO PARADIGMA: Input → Variazione Creativa
    # NON più denoising (noisy → clean), ma generazione (clean → variation)
    orphans_by_pred = defaultdict(list)
    for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
        if isinstance(o, Literal):
            val = str(o).strip()
            if val:
                p_name = clean_p(p, attr_map).replace('_', ' ')
                orphans_by_pred[p_name].append(val)

    orphan_count = 0
    for p_name, vals in orphans_by_pred.items():
        unique_vals = list(set(vals))
        # Ridotto: max 100 orphans per predicato (era 200)
        selected = random.sample(unique_vals, min(len(unique_vals), 100))
        for v in selected:
            # Solo 50% degli orphans (riduciamo peso sintetico)
            if random.random() < 0.5:
                v_creative = creative_gen.generate(v, predicate=p_name)
                # Verifica che sia sufficientemente diverso
                if v_creative != v and len(v_creative) > 2:
                    t5_rows.append({"input": f"generate variation <{p_name}>: {v}", "target": v_creative})
                    orphan_count += 1

    print(f"    [ORPHANS] Synthetic variations: {orphan_count}")

    # FILTRI QUALITÀ: rimuovi unicode garbage e token swap
    print(f"    Pre-filter samples: {len(t5_rows)}")
    t5_rows = filter_training_data(t5_rows)

    random.shuffle(t5_rows)
    print(f"    Total training samples: {len(t5_rows)} (Creative Variation paradigm, filtered)")

    # 2. MODEL XL (BF16 + LoRA)
    device = "cuda"
    out_dir = "./results/t5_xl_creative_v6"  # v6: più peso a coppie reali, meno sintetiche
    interpolator = MixupT5XLInterpolator(model_name=MODEL_NAME, out_dir=out_dir, device=device)

    # 3. TRAINING (Forza retraining per nuovo paradigma)
    if not (Path(out_dir) / "adapter_model.bin").exists() and not (Path(out_dir) / "adapter_model.safetensors").exists():
        print(f"    [2/4] Fine-tuning with CREATIVE VARIATION paradigm...")
        interpolator.fine_tune(t5_rows, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3)
    else: print("    [2/4] XL Creative Adapters found.")

    # 3. SWEEP (Range espansi per variazioni più creative)
    print(f"\n>>> PHASE 3: SWEEPING (Extended ranges)")
    sweep_pool = []
    all_test_preds = list(aligned_test_pool.keys())
    for _ in range(min(SWEEP_SAMPLES, len(all_test_preds))):
        p = random.choice(all_test_preds)
        v1, v2 = random.choice(aligned_test_pool[p])
        sweep_pool.append((p, v1, v2))

    sweep_results = []
    for a in [0.3, 0.4, 0.5]:  # Alpha moderati (v3-style)
        for n in [0.0, 0.01, 0.02]: # Rumore latente basso
            for t in [0.7, 0.85, 1.0]:  # Temperature conservative
                interpolator.latent_noise_std, interpolator.gen_temperature = n, t
                scs = []
                for p, v1, v2 in sweep_pool:
                    aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=a)
                    res = calculate_pair_score(v1, v2, aa, ab)
                    scs.append(res['score'])
                avg = np.mean(scs)
                sweep_results.append({"a": a, "n": n, "t": t, "score": avg})
                print(f"      - Alpha={a} Noise={n} Temp={t} -> Score: {avg:.3f}")
    
    best = sorted(sweep_results, key=lambda x: x['score'], reverse=True)[0]
    print(f"    BEST CONFIG: {best}")

    # 4. REPORT
    print(f"    [4/4] Generating Creative Variation Report...")
    interpolator.latent_noise_std, interpolator.gen_temperature = best['n'], best['t']
    output_file = "massive_t5_xl_creative_v6_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DAKGEA FLAN-T5-XL (3B) CREATIVE VARIATION REPORT v6\n")
        f.write(f"Config: {best} | Model: XL | Strategy: v3-restored (alpha<=0.5, temp<=1.0)\n")
        f.write("="*120 + "\n\n")
        
        # Aligned
        f.write("SECTION 1: ALIGNED MIXUP (XL)\n" + "-"*80 + "\n")
        p_names = sorted(list(aligned_test_pool.keys()))
        count = 0
        while count < TOTAL_REPORT_SAMPLES // 2 and p_names:
            for p in p_names[:]:
                if not aligned_test_pool[p]: p_names.remove(p); continue
                v1, v2 = aligned_test_pool[p].pop(random.randrange(len(aligned_test_pool[p])))
                aa, ab = interpolator.interpolate_pair(v1, v2, predicate=p, alpha=best['a'])
                res = calculate_pair_score(v1, v2, aa, ab)
                f.write(f"PAIR {count+1:03d} | {p:20} | V1: {v1[:35]:35} -> AUG: {aa[:35]:35}\n")
                f.write(f"         | Score: {res['score']:.2f}        | V2: {v2[:35]:35} -> AUG: {ab[:35]:35} | Voto:[ ]/5\n")
                f.write("-" * 120 + "\n")
                count += 1
                if count >= TOTAL_REPORT_SAMPLES // 2: break

        # Orphans
        f.write("\nSECTION 2: ORPHAN VARIATIONS (XL)\n" + "-"*80 + "\n")
        o_names = sorted(list(orphans_by_pred.keys()))
        o_count = 0
        while o_count < TOTAL_REPORT_SAMPLES // 2 and o_names:
            for p in o_names[:]:
                if not orphans_by_pred[p]: o_names.remove(p); continue
                val = orphans_by_pred[p].pop(random.randrange(len(orphans_by_pred[p])))
                # Per generare una variazione, usiamo il modello allenato al denoising
                # ma con temperatura > 0, che lo forza a non essere puramente conservativo.
                aa, _ = interpolator.interpolate_pair(val, val, predicate=p, alpha=0.5)
                score = calculate_score(val, aa)
                f.write(f"ORPHAN {o_count+1:03d} | {p:20} | ORIG: {val[:45]:45} -> VAR: {aa[:45]:45} (Score: {score:.2f}) | Voto:[ ]/5\n")
                o_count += 1
                if o_count >= TOTAL_REPORT_SAMPLES // 2: break

    print(f"\n>>> SUCCESS: XL Creative v6 Report (v3-restored) saved to {output_file}")

if __name__ == "__main__":
    run_xl_pipeline()
