import sys
import torch
import logging
import random
import time
import numpy as np
import re
import os
import argparse
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

# Available datasets
AVAILABLE_DATASETS = ["BBC_DB", "D_W_15K_V1", "D_W_15K_V2", "ICEW_WIKI", "ICEW_YAGO"]

logger = get_logger("FlanT5XL_Creative")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
creative_gen = CreativeVariationGenerator()

# --- FILTRI QUALITÀ TRAINING DATA ---
# Pattern più ampi per catturare vari formati di escape unicode
UNICODE_GARBAGE_PATTERNS = [
    re.compile(r'u00[a-f0-9]{2}', re.IGNORECASE),      # u00e9, u00f3
    re.compile(r'\\u00[a-f0-9]{2}', re.IGNORECASE),   # \u00e9
    re.compile(r'&#x[a-f0-9]{2,4};', re.IGNORECASE),  # &#xe9;
    re.compile(r'&#\d{2,4};'),                         # &#233;
    re.compile(r'%[a-f0-9]{2}', re.IGNORECASE),       # %e9 (URL encoding)
]
FILLER_WORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for'}

def fix_unicode_escapes(text: str) -> str:
    """
    Prova a decodificare escape unicode comuni.
    u00e9 → é, u00f3 → ó, etc.
    """
    import codecs

    result = text

    # Pattern: u00XX (senza backslash)
    def replace_u00(match):
        try:
            hex_val = match.group(0)[1:]  # rimuovi 'u'
            return chr(int(hex_val, 16))
        except:
            return match.group(0)

    result = re.sub(r'u00[a-f0-9]{2}', replace_u00, result, flags=re.IGNORECASE)

    # Pattern: \uXXXX (con backslash)
    try:
        result = codecs.decode(result, 'unicode_escape')
    except:
        pass

    return result

def has_unicode_garbage(text: str) -> bool:
    """Rileva caratteri unicode corrotti come u00e9, u00f3, etc."""
    return any(p.search(text) for p in UNICODE_GARBAGE_PATTERNS)

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

# Coppie di FLIP da insegnare a T5
FLIP_PAIRS = {
    'yes': 'no', 'no': 'yes',
    'true': 'false', 'false': 'true',
    'left': 'right', 'right': 'left',
    'up': 'down', 'down': 'up',
    'top': 'bottom', 'bottom': 'top',
    'start': 'end', 'end': 'start',
    'on': 'off', 'off': 'on',
    'active': 'inactive', 'inactive': 'active',
    'enabled': 'disabled', 'disabled': 'enabled',
}

def generate_flip_training_pairs() -> list:
    """
    Genera coppie di training per FLIP + IDENTITY.

    T5 impara sia a flippare (yes→no) che a mantenere (yes→yes).
    Bilanciato 50/50 per evitare che flippi sempre!
    """
    pairs = []
    for val, flipped in FLIP_PAIRS.items():
        # 50% FLIP: yes → no
        for _ in range(3):
            pairs.append({"input": f"generate variation <value>: {val}", "target": flipped})

        # 50% IDENTITY: yes → yes (a volte non cambiare!)
        for _ in range(3):
            pairs.append({"input": f"generate variation <value>: {val}", "target": val})

    return pairs

# --- VARIAZIONI MULTI-WORD (puramente algoritmiche, scalabili) ---

def has_vowel(word: str) -> bool:
    """Controlla se una parola ha almeno una vocale."""
    return any(c.lower() in 'aeiou' for c in word)

def vary_word_algorithmic(word: str) -> str:
    """
    Varia una parola usando SOLO algoritmi scalabili (NO dizionari!).
    GARANTISCE che il risultato sia pronunciabile (almeno 1 vocale).

    Tecniche v14 (scalabili su qualsiasi dataset):
    1. Suffissi universali: -y, -ie, -son, -sen, -man, -er, -ini, -elli
    2. Prefissi titoli: jr-, mr-, dr-, st-
    3. Troncamento intelligente: prime 3-4 lettere (Alex da Alexander)
    4. Abbreviazione iniziale: J. Smith, R. Johnson
    5. Espansione vocale: steve → steeve (raddoppio vocale, non consonante!)
    6. Swap vocali: a↔e, i↔o (più naturale di consonanti)
    """
    if len(word) < 2:
        return word

    # v14: Tecniche scalabili, NO dizionario
    technique = random.choice([
        'suffix', 'suffix', 'suffix',    # 3x peso - molto naturale
        'prefix',                         # 1x
        'truncate', 'truncate',          # 2x peso - crea diminutivi naturali
        'initial',                        # 1x - abbreviazioni (J., R.)
        'vowel_double',                   # 1x - più naturale di consonante
        'vowel_swap'                      # 1x - scambio vocali
    ])

    result = word  # Default

    if technique == 'suffix':
        # Suffissi universali (funzionano in molte lingue)
        suffix = random.choice(['y', 'ie', 'son', 'sen', 'man', 'er', 'ini', 'elli', 'ski', 'ov'])
        # Evita doppie: smithy non smithyy
        if word.endswith(suffix[0]):
            result = word + suffix[1:] if len(suffix) > 1 else word + suffix
        else:
            result = word + suffix

    elif technique == 'prefix':
        # Prefissi titolo (universali)
        prefix = random.choice(['jr', 'mr', 'dr', 'st', 'von', 'de', 'van'])
        result = prefix + word

    elif technique == 'truncate' and len(word) >= 4:
        # Troncamento intelligente: crea diminutivi naturali
        # alexander → alex, elizabeth → eliz, robert → rob
        # Prendi le prime 3-4 lettere se contengono almeno una vocale
        for length in [4, 3]:
            if len(word) > length:
                candidate = word[:length]
                if has_vowel(candidate):
                    result = candidate
                    break
        # Fallback: tronca 1-2 caratteri dalla fine
        if result == word:
            for cut in range(1, min(3, len(word) - 2)):
                candidate = word[:-cut]
                if has_vowel(candidate) and len(candidate) >= 2:
                    result = candidate
                    break

    elif technique == 'initial':
        # Abbreviazione iniziale: Robert → R., John Smith → J. Smith
        # Solo per parole lunghe (>3 caratteri)
        if len(word) > 3:
            result = word[0].upper() + '.'

    elif technique == 'vowel_double':
        # Raddoppia una VOCALE (più naturale di consonante)
        # steve → steeve, john → joohn
        vowels = 'aeiou'
        vowel_positions = [i for i, c in enumerate(word.lower()) if c in vowels]
        if vowel_positions:
            idx = random.choice(vowel_positions)
            result = word[:idx] + word[idx] + word[idx:]

    elif technique == 'vowel_swap':
        # Scambia una vocale con un'altra vicina (più naturale)
        # steve → stave, john → jahn
        vowels = 'aeiou'
        vowel_positions = [i for i, c in enumerate(word.lower()) if c in vowels]
        if vowel_positions:
            idx = random.choice(vowel_positions)
            old_vowel = word[idx].lower()
            new_vowel = random.choice([v for v in vowels if v != old_vowel])
            result = word[:idx] + new_vowel + word[idx+1:]

    # SAFETY CHECK: se risultato non ha vocali, usa fallback sicuro
    if not has_vowel(result):
        result = word + random.choice(['y', 'a', 'o'])  # Aggiungi vocale

    return result

# v14: LEARNED VARIATIONS - estratte direttamente dalle coppie allineate!
LEARNED_VARIATIONS = {}  # Popolato da learn_variations_from_pairs()

def learn_variations_from_pairs(aligned_pairs: list) -> dict:
    """
    Impara variazioni di parole direttamente dalle coppie allineate.

    Input: [("Bob Smith", "Robert Smith"), ("Mike Johnson", "Michael Johnson"), ...]
    Output: {"bob": ["robert"], "robert": ["bob"], "mike": ["michael"], ...}

    Questo è SCALABILE perché:
    - Non richiede dizionari manuali
    - Impara dal dataset stesso
    - Cattura variazioni domain-specific
    """
    from difflib import SequenceMatcher

    variations = {}

    for val_src, val_tgt in aligned_pairs:
        words_src = val_src.lower().split()
        words_tgt = val_tgt.lower().split()

        # Solo se stesso numero di parole (allineamento 1:1)
        if len(words_src) != len(words_tgt):
            continue

        # Trova parole diverse nella stessa posizione
        for ws, wt in zip(words_src, words_tgt):
            if ws == wt:
                continue  # Stessa parola, skip

            # Verifica che siano "simili" (non completamente diverse)
            # Similarità > 0.4 significa che condividono qualche carattere
            sim = SequenceMatcher(None, ws, wt).ratio()
            if sim < 0.3 or sim > 0.95:
                continue  # Troppo diverse o troppo simili

            # Aggiungi la variazione bidirezionale
            if ws not in variations:
                variations[ws] = set()
            if wt not in variations:
                variations[wt] = set()

            variations[ws].add(wt)
            variations[wt].add(ws)

    # Converti set in liste
    return {k: list(v) for k, v in variations.items() if v}

def vary_word_with_learned(word: str, learned: dict) -> str:
    """
    Varia una parola usando prima le variazioni APPRESE, poi algoritmi.

    Priorità:
    1. Se la parola è nel dizionario appreso → usa variazione appresa (70%)
    2. Altrimenti → usa algoritmo
    """
    word_lower = word.lower()

    # v14: Prima prova variazioni apprese dal dataset
    if word_lower in learned and learned[word_lower]:
        if random.random() < 0.7:  # 70% usa variazione appresa
            variant = random.choice(learned[word_lower])
            # Mantieni capitalizzazione
            if word[0].isupper():
                variant = variant.capitalize()
            return variant

    # Fallback: usa algoritmo
    return vary_word_algorithmic(word)

def vary_all_words(text: str, learned: dict = None) -> str:
    """Varia OGNI parola usando variazioni apprese + algoritmi."""
    if learned is None:
        learned = LEARNED_VARIATIONS

    words = text.split()
    if len(words) < 2:
        return vary_word_with_learned(text, learned) if learned else vary_word_algorithmic(text)

    # Varia OGNI parola!
    if learned:
        varied_words = [vary_word_with_learned(w, learned) for w in words]
    else:
        varied_words = [vary_word_algorithmic(w) for w in words]
    return ' '.join(varied_words)

def generate_multi_word_training_pairs(names: list, n_per_name: int = 3, learned: dict = None) -> list:
    """
    Genera coppie di training dove OGNI parola viene modificata.
    v14: Usa variazioni APPRESE dal dataset + algoritmi come fallback.

    Input: ["john smith", "steve marriott", ...]
    Output: [
        {"input": "generate variation <name>: john smith", "target": "jonathan smithson"},
        {"input": "generate variation <name>: steve marriott", "target": "steven marriot"},
        ...
    ]
    """
    if learned is None:
        learned = LEARNED_VARIATIONS

    pairs = []
    for name in names:
        words = name.split()
        if len(words) < 2:
            continue  # Skip single-word names

        for _ in range(n_per_name):
            varied = vary_all_words(name, learned)  # v14: usa variazioni apprese!
            # Assicurati che sia effettivamente diverso
            if varied.lower() != name.lower():
                pairs.append({
                    "input": f"generate variation <name>: {name}",
                    "target": varied
                })

    return pairs

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

        # v14: Prima prova a DECODIFICARE unicode escapes
        tgt_fixed = fix_unicode_escapes(tgt)
        if tgt_fixed != tgt:
            tgt = tgt_fixed
            row['target'] = tgt_fixed  # Aggiorna anche il row

        # Estrai valore dall'input per confronto
        inp_val = extract_value_from_prompt(inp)

        # Fix 1: Rimuovi unicode garbage (dopo tentativo di fix)
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

    print(f"    [FILTER] Removed: {unicode_removed} unicode, {swap_removed} swaps, {filler_removed} filler, {too_similar_removed} similar")
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

def build_canonical_map_from_matches(dataset) -> dict:
    """Build canonical map using attribute_matches (same logic as MixupDataBuilder)."""
    canonical_map = {}

    # First, map all predicates from attribute_matches
    if dataset.attribute_matches:
        for src_uri, tgt_uris in dataset.attribute_matches.items():
            local = src_uri.split("/")[-1].split("#")[-1]
            token = f"<{local.upper()}>"
            canonical_map[src_uri] = token
            for tgt_uri in tgt_uris:
                canonical_map[tgt_uri] = token
        print(f"    [CANONICAL] Built from attribute_matches: {len(canonical_map)} URIs -> {len(set(canonical_map.values()))} tokens")
    else:
        print("    [WARN] No attribute_matches available!")

    # Add remaining predicates
    all_predicates = (set(dataset.knowledge_graph_source.predicates()) |
                     set(dataset.knowledge_graph_target.predicates()))
    for p in all_predicates:
        p_str = str(p)
        if p_str not in canonical_map:
            local = p_str.split("/")[-1].split("#")[-1].upper()
            canonical_map[p_str] = f"<{local}>"

    return canonical_map


def run_xl_pipeline(dataset_name: str = "BBC_DB"):
    print("\n" + "█"*100); print(f"█ RTX 4090: FLAN-T5 XL (3B) - CREATIVE VARIATION PIPELINE [{dataset_name}] ".center(98) + "█"); print("█"*100)

    # 1. DATA
    dataset_path = str(PROJECT_ROOT / "data/raw/openea" / dataset_name)
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return
    reader = OpeneaDatasetReader()
    dataset = reader.read(dataset_path)
    attr_map = load_attr_names(dataset_path)

    # Build canonical map using attribute_matches (key fix for non-BBC_DB datasets!)
    canonical_map = build_canonical_map_from_matches(dataset)

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

    # v14: IMPARA VARIAZIONI dalle coppie allineate!
    all_aligned_pairs = []
    for pairs_list in aligned_test_pool.values():
        all_aligned_pairs.extend(pairs_list)

    global LEARNED_VARIATIONS
    LEARNED_VARIATIONS = learn_variations_from_pairs(all_aligned_pairs)
    print(f"    [LEARNED] Extracted {len(LEARNED_VARIATIONS)} word variations from aligned pairs")

    # Mostra alcuni esempi di variazioni apprese
    if LEARNED_VARIATIONS:
        examples = list(LEARNED_VARIATIONS.items())[:5]
        for word, variants in examples:
            print(f"      - {word} ↔ {variants}")

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

    # AGGIUNGI FLIP PAIRS: yes↔no, true↔false, left↔right, etc.
    flip_pairs = generate_flip_training_pairs()
    t5_rows.extend(flip_pairs)
    print(f"    [FLIP] Added {len(flip_pairs)} flip training pairs (yes↔no, left↔right, etc.)")

    # AGGIUNGI MULTI-WORD VARIATIONS: "john smith" → "johnny smyth" (OGNI parola cambia!)
    # Raccogli nomi multi-word dal training
    multi_word_names = set()
    for row in t5_rows:
        inp = row['input']
        if '<name>' in inp.lower():
            val = extract_value_from_prompt(inp)
            if len(val.split()) >= 2 and len(val) < 50:  # Solo nomi 2+ parole, non troppo lunghi
                multi_word_names.add(val)

    # Genera variazioni dove OGNI parola viene modificata
    multi_word_pairs = generate_multi_word_training_pairs(list(multi_word_names), n_per_name=10)  # 10x peso!
    t5_rows.extend(multi_word_pairs)
    print(f"    [MULTI-WORD] Added {len(multi_word_pairs)} pairs from {len(multi_word_names)} names (ogni parola varia!)")

    random.shuffle(t5_rows)
    print(f"    Total training samples: {len(t5_rows)} (Creative + flips + multi-word)")

    # 2. MODEL XL (BF16 + LoRA)
    device = "cuda"
    out_dir = f"./results/t5_xl_creative_v14_{dataset_name}"  # Separate model per dataset
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
            for t in [0.7]:  # v11: FISSO a 0.7 (v9 funzionava, v10 con 1.0 era garbage)
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
    output_file = f"massive_t5_xl_creative_v14_{dataset_name}_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"DAKGEA FLAN-T5-XL (3B) CREATIVE VARIATION REPORT v14 [{dataset_name}]\n")
        f.write(f"Config: {best} | Model: XL | Strategy: v14 - NAME VARIANTS + less typos\n")
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

    print(f"\n>>> SUCCESS: XL Creative v14 Report (NAME VARIANTS + less typos) saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAN-T5 XL Creative Variation Pipeline")
    parser.add_argument("--dataset", "-d", type=str, default="BBC_DB",
                       choices=AVAILABLE_DATASETS,
                       help=f"Dataset to use (default: BBC_DB). Options: {AVAILABLE_DATASETS}")
    args = parser.parse_args()
    run_xl_pipeline(args.dataset)
