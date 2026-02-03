"""
Scoring module per valutare la qualità degli output del Mix-up.

=== DUE MODALITÀ ===

1. TESTI CORTI (≤5 parole): nomi, date, ID
   - Vogliamo MIX: elementi da entrambi gli input
   - Connessione: parole simili (John~Jonathan)
   - Abbreviazioni: J. = John

2. TESTI LUNGHI (>5 parole): commenti, descrizioni
   - OK summarization, cambio stile
   - Conta la diversità ma non il word matching esatto

=== PRINCIPI ===
- IDENTITY (score=0): penalizzata sempre
- GARBAGE (score=0.1): nessuna connessione semantica
- VARIATION (score=0.6-0.8): diverso ma connesso
- MIX (score=1.0): elementi da entrambi gli input
"""

import re
from difflib import SequenceMatcher
from typing import Dict, Tuple, Set

# Soglia per distinguere testi corti da lunghi
SHORT_TEXT_THRESHOLD = 5

# Articoli e parole "filler" che non contano come variazione creativa
FILLER_WORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for'}

# Suffissi pigri che da soli non contano come variazione creativa
LAZY_SUFFIXES = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv'}

# Pattern per rilevare garbage unicode (es: u00e9l, u00f3n)
UNICODE_GARBAGE_PATTERN = re.compile(r'u00[a-f0-9]{2}', re.IGNORECASE)

# Semantic model per testi lunghi (lazy loading)
_semantic_model = None

def _get_semantic_model():
    """Lazy loading del modello semantico."""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            _semantic_model = False  # Fallback: non disponibile
    return _semantic_model if _semantic_model else None


def _semantic_similarity(text1: str, text2: str) -> float:
    """Calcola similarità semantica tra due testi."""
    model = _get_semantic_model()
    if model is None:
        # Fallback a word overlap se modello non disponibile
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / max(len(words1), len(words2))

    from sentence_transformers import util
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def _word_sim(w1: str, w2: str) -> float:
    """Similarità tra due parole."""
    return SequenceMatcher(None, w1.lower(), w2.lower()).ratio()


def _has_vowel(word: str) -> bool:
    """Controlla se una parola ha almeno una vocale (pronunciabile)."""
    # Include 'y' che spesso funge da vocale (Smyth, Lynn, etc.)
    return any(c.lower() in 'aeiouy' for c in word)


def _is_only_lazy_suffix(orig: str, gen: str) -> bool:
    """
    Controlla se l'unica modifica è l'aggiunta di un suffisso pigro (jr, sr, etc).

    "john smith" → "john smith jr" = True (SOLO jr aggiunto)
    "john smith" → "jhon smith jr" = False (anche typo!)
    """
    orig_words = orig.lower().split()
    gen_words = gen.lower().split()

    # Deve avere esattamente 1 parola in più
    if len(gen_words) != len(orig_words) + 1:
        return False

    # L'ultima parola deve essere un suffisso pigro
    if gen_words[-1] not in LAZY_SUFFIXES:
        return False

    # Le altre parole devono essere IDENTICHE
    for ow, gw in zip(orig_words, gen_words[:-1]):
        if ow != gw:
            return False  # C'è anche un'altra modifica, OK!

    return True  # Solo suffisso pigro, penalizza!


def _count_unpronounceable_words(text: str) -> int:
    """
    Conta parole senza vocali (impronunciabili).

    Esclude:
    - Parole corte (<=2 chars): spesso abbreviazioni (jr, mr, dr)
    - Abbreviazioni comuni note
    """
    # Abbreviazioni comuni senza vocali
    COMMON_ABBREVS = {'jr', 'sr', 'mr', 'dr', 'st', 'nd', 'rd', 'th'}

    words = re.findall(r'\w+', text.lower())
    count = 0
    for w in words:
        # Salta parole corte e abbreviazioni
        if len(w) <= 2 or w in COMMON_ABBREVS:
            continue
        if not _has_vowel(w):
            count += 1
    return count


def _count_modified_words(orig: str, gen: str) -> tuple:
    """
    Conta quante parole sono state modificate.

    Returns: (modified_count, total_words, all_modified)
    """
    orig_words = orig.lower().split()
    gen_words = gen.lower().split()

    # Se numero parole diverso, conta come "tutte modificate"
    if len(orig_words) != len(gen_words):
        return len(gen_words), len(gen_words), True

    modified = 0
    for ow, gw in zip(orig_words, gen_words):
        if ow != gw:
            modified += 1

    return modified, len(orig_words), modified == len(orig_words)


def _is_abbreviation(short: str, full: str) -> bool:
    """Verifica se 'short' è un'abbreviazione di 'full'. Es: J -> John"""
    if len(short) == 1:
        return full.lower().startswith(short.lower())
    if len(short) <= 3 and len(full) > 3:
        return full.lower().startswith(short.lower())
    return False


def _is_char_swap(word1: str, word2: str) -> bool:
    """
    Rileva se due parole differiscono solo per uno swap di caratteri adiacenti.

    "hidalgo" vs "hdialgo" → True (i e d swappati)
    "smith" vs "smtih" → True (i e t swappati)
    "john" vs "johnny" → False (lunghezze diverse)
    """
    w1, w2 = word1.lower(), word2.lower()

    # Lunghezze devono essere uguali
    if len(w1) != len(w2):
        return False

    # Trova le posizioni diverse
    diffs = [(i, c1, c2) for i, (c1, c2) in enumerate(zip(w1, w2)) if c1 != c2]

    # Per uno swap, ci aspettiamo esattamente 2 differenze adiacenti con caratteri invertiti
    if len(diffs) == 2:
        i1, c1_at_1, c2_at_1 = diffs[0]
        i2, c1_at_2, c2_at_2 = diffs[1]
        # Devono essere adiacenti e i caratteri invertiti
        if abs(i1 - i2) == 1 and c1_at_1 == c2_at_2 and c1_at_2 == c2_at_1:
            return True

    return False


def _count_char_swaps(gen: str, orig: str) -> int:
    """
    Conta quante parole nell'output sono char swaps rispetto all'input.

    "david hdialgo" vs "david hidalgo" → 1 (hdialgo è swap di hidalgo)
    "jhon smtih" vs "john smith" → 2 (entrambe swappate)
    """
    gen_words = gen.lower().split()
    orig_words = orig.lower().split()

    if len(gen_words) != len(orig_words):
        return 0

    swaps = 0
    for gw, ow in zip(gen_words, orig_words):
        if gw == ow:
            continue  # Identico, non conta
        if _is_char_swap(gw, ow):
            swaps += 1
        elif _word_sim(gw, ow) < 0.7:
            # Parola completamente diversa, non è un caso di char swap
            return 0

    return swaps


def _find_connection(word: str, word_set: Set[str]) -> Tuple[str, float, str]:
    """
    Trova la migliore connessione tra una parola e un set.

    Returns:
        (matched_word, score, connection_type)
        connection_type: "exact", "abbreviation", "similar", "none"
    """
    word_l = word.lower()

    # 1. Match esatto
    if word_l in {w.lower() for w in word_set}:
        return word, 1.0, "exact"

    # 2. Abbreviazione (J -> John)
    for w in word_set:
        if _is_abbreviation(word_l, w):
            return w, 0.7, "abbreviation"

    # 3. Similarità
    best_match = None
    best_sim = 0.0
    for w in word_set:
        if len(w) < 2:
            continue
        sim = _word_sim(word_l, w)
        if sim > best_sim:
            best_sim = sim
            best_match = w

    if best_sim >= 0.5:
        return best_match, best_sim, "similar"

    return None, 0.0, "none"


def _score_short_text(orig: str, gen: str, other: str) -> Dict:
    """Scoring per testi corti (nomi, date, ID)."""

    gen_c = gen.strip().lower()
    orig_c = orig.strip().lower()
    other_c = other.strip().lower() if other else ""

    # === PENALITÀ FORTI (prima di tutto) ===

    # Garbage unicode check (es: u00e9l, u00f3n)
    unicode_matches = UNICODE_GARBAGE_PATTERN.findall(gen_c)
    if len(unicode_matches) >= 2:
        return {"score": 0.05, "reason": "unicode_garbage", "matches": unicode_matches}

    # PAROLE IMPRONUNCIABILI (senza vocali) = penalità forte (-3 equivale a score 0.0)
    unpronounceable = _count_unpronounceable_words(gen_c)
    if unpronounceable > 0:
        return {"score": 0.0, "reason": "unpronounceable",
                "unpronounceable_count": unpronounceable, "text": gen_c}

    # Identity check
    if gen_c == orig_c or gen_c == other_c:
        return {"score": 0.0, "reason": "identity"}

    # SOLO SUFFISSO PIGRO (jr, sr) senza altre modifiche = penalità (-2 equivale a score 0.1)
    if _is_only_lazy_suffix(orig_c, gen_c):
        return {"score": 0.1, "reason": "only_lazy_suffix",
                "suffix": gen_c.split()[-1], "note": "jr/sr alone is not creative"}

    # === BONUS: Multi-word con TUTTE le parole modificate (check PRIMA di char_swap) ===
    modified_A, total_A, all_modified_A = _count_modified_words(orig_c, gen_c)
    if total_A >= 2 and all_modified_A:
        # TUTTE le parole modificate in un multi-word = ECCELLENTE!
        return {"score": 1.0, "reason": "all_words_modified",
                "modified": modified_A, "total": total_A,
                "note": "every word changed - excellent variation!"}

    # CHAR SWAP check - DOPO all_words_modified!
    # "david hdialgo" vs "david hidalgo" → char swap valido, non near-copy!
    char_swaps_A = _count_char_swaps(gen_c, orig_c)
    char_swaps_B = _count_char_swaps(gen_c, other_c) if other_c else 0

    if char_swaps_A > 0 or char_swaps_B > 0:
        total_swaps = max(char_swaps_A, char_swaps_B)
        # Calcola quante parole ci sono in totale
        n_words = len(gen_c.split())

        if n_words == 1:
            # Parola singola con char swap → 0.70
            score = 0.70
        elif total_swaps >= n_words:
            # TUTTE le parole variate → 1.00 (perfetto!)
            score = 1.00
        else:
            # Variazione parziale: 1/2 → 0.40, 2/3 → 0.47, etc.
            # Formula: 0.20 + (ratio) * 0.40
            ratio = total_swaps / n_words
            score = 0.20 + ratio * 0.40

        return {"score": round(score, 2), "reason": "char_swap_variation",
                "swaps": total_swaps, "total_words": n_words}

    # Near-copy check (solo se NON è un char swap)
    sim_A = _word_sim(gen_c, orig_c)
    sim_B = _word_sim(gen_c, other_c) if other_c else 0.0

    if sim_A > 0.9 or sim_B > 0.9:
        return {"score": 0.2, "reason": "near_copy", "sim": max(sim_A, sim_B)}

    # Analisi parole
    gen_words = set(re.findall(r'\w+', gen_c))
    words_A = set(re.findall(r'\w+', orig_c))
    words_B = set(re.findall(r'\w+', other_c)) if other_c else set()
    all_input_words = words_A | words_B

    # Check "solo aggiunta filler" (the, and, a...)
    # Se l'unica differenza è un filler aggiunto/rimosso, penalizza
    diff_added = gen_words - all_input_words
    diff_removed = all_input_words - gen_words

    # Caso: aggiunto solo filler (es: "rollins band" -> "rollins the band")
    if diff_added and diff_added <= FILLER_WORDS and not diff_removed:
        return {"score": 0.15, "reason": "only_filler_added", "filler": list(diff_added)}

    # Caso: rimosso solo contenuto, aggiunto filler (es: "X Y Z" -> "X the")
    meaningful_added = diff_added - FILLER_WORDS
    if not meaningful_added and diff_removed:
        return {"score": 0.15, "reason": "removed_content_added_filler",
                "removed": list(diff_removed), "filler": list(diff_added & FILLER_WORDS)}

    # Shuffle check: output usa SOLO token dagli input (nessuna creatività)
    if len(gen_words) > 1 and gen_words <= all_input_words:
        # Output è solo riordino/subset dei token input
        return {"score": 0.15, "reason": "shuffle", "tokens": list(gen_words)}

    # Input identici?
    inputs_identical = orig_c == other_c if other_c else False

    # Trova connessioni
    connections_A = []  # [(gen_word, matched, score, type)]
    connections_B = []
    unconnected = []

    for gw in gen_words:
        conn_A = _find_connection(gw, words_A)
        conn_B = _find_connection(gw, words_B)

        # Se connesso a ENTRAMBI con score simili (diff < 0.15), conta per entrambi
        if conn_A[1] > 0.4 and conn_B[1] > 0.4:
            diff = abs(conn_A[1] - conn_B[1])
            if diff < 0.15:
                # Connesso a entrambi = MIX
                connections_A.append((gw, *conn_A))
                connections_B.append((gw, *conn_B))
            elif conn_A[1] > conn_B[1]:
                connections_A.append((gw, *conn_A))
            else:
                connections_B.append((gw, *conn_B))
        elif conn_A[1] > 0.4:
            connections_A.append((gw, *conn_A))
        elif conn_B[1] > 0.4:
            connections_B.append((gw, *conn_B))
        else:
            unconnected.append(gw)

    has_A = len(connections_A) > 0
    has_B = len(connections_B) > 0
    has_mix = has_A and has_B and not inputs_identical

    # Scoring
    if has_mix:
        return {
            "score": 1.0,
            "reason": "perfect_mix",
            "from_A": connections_A,
            "from_B": connections_B,
        }

    if has_A or has_B:
        if inputs_identical:
            return {
                "score": 0.8,
                "reason": "good_variation_identical_inputs",
                "connections": connections_A or connections_B,
            }
        return {
            "score": 0.7,  # Alzato da 0.6
            "reason": "partial_connection",
            "connections": connections_A or connections_B,
        }

    # Nessuna connessione
    return {"score": 0.1, "reason": "garbage", "unconnected": unconnected}


def _score_long_text(orig: str, gen: str, other: str) -> Dict:
    """
    Scoring per testi lunghi (commenti, descrizioni).

    Usa SEMANTIC SIMILARITY per valutare:
    - Riformulazione (stesso significato, parole diverse) → alto
    - Summarization (più corto ma semantica OK) → alto
    - Cambio stile (British→American) → alto
    - Espansione (aggiunge dettagli coerenti) → alto
    """
    gen_c = gen.strip()
    orig_c = orig.strip()
    other_c = other.strip() if other else ""

    # Garbage unicode check (es: u00e9l, u00f3n)
    unicode_matches = UNICODE_GARBAGE_PATTERN.findall(gen_c.lower())
    if len(unicode_matches) >= 2:
        return {"score": 0.05, "reason": "unicode_garbage", "matches": unicode_matches}

    # Identity check (lessicale)
    lex_sim_A = _word_sim(gen_c.lower(), orig_c.lower())
    lex_sim_B = _word_sim(gen_c.lower(), other_c.lower()) if other_c else 0.0

    if lex_sim_A > 0.95 or lex_sim_B > 0.95:
        return {"score": 0.0, "reason": "identity"}

    if lex_sim_A > 0.90 or lex_sim_B > 0.90:
        return {"score": 0.2, "reason": "near_copy"}

    # Semantic similarity
    sem_sim_A = _semantic_similarity(gen_c, orig_c)
    sem_sim_B = _semantic_similarity(gen_c, other_c) if other_c else 0.0
    max_sem_sim = max(sem_sim_A, sem_sim_B)

    # Scoring basato su semantica
    # Vogliamo: diverso lessicalmente MA simile semanticamente

    max_lex_sim = max(lex_sim_A, lex_sim_B)

    if max_sem_sim > 0.75:
        # Buona similarità semantica
        if max_lex_sim < 0.7:
            # Diverso lessicalmente ma semanticamente simile = OTTIMO
            return {
                "score": 1.0,
                "reason": "excellent_rephrase",
                "semantic_sim": max_sem_sim,
                "lexical_sim": max_lex_sim
            }
        else:
            # Troppo simile lessicalmente
            return {
                "score": 0.6,
                "reason": "good_but_similar",
                "semantic_sim": max_sem_sim,
                "lexical_sim": max_lex_sim
            }

    if max_sem_sim > 0.5:
        # Semantica parziale - potrebbe essere summarization
        return {
            "score": 0.7,
            "reason": "partial_semantic_match",
            "semantic_sim": max_sem_sim,
            "note": "possible summarization or style change"
        }

    if max_sem_sim > 0.3:
        # Bassa semantica ma non zero
        return {
            "score": 0.4,
            "reason": "weak_semantic_connection",
            "semantic_sim": max_sem_sim
        }

    # Nessuna connessione semantica = garbage
    return {
        "score": 0.1,
        "reason": "garbage",
        "semantic_sim": max_sem_sim
    }


def _find_shared_tokens(text1: str, text2: str) -> Set[str]:
    """Trova token condivisi (o simili) tra due testi."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))

    shared = words1 & words2  # Esatti

    # Aggiungi anche parole simili (>0.7)
    for w1 in words1 - shared:
        for w2 in words2 - shared:
            if _word_sim(w1, w2) > 0.7:
                shared.add(w1)
                break

    return shared


def _coherence_bonus(input_A: str, input_B: str, output_A: str, output_B: str) -> Tuple[float, Dict]:
    """
    Calcola bonus di coerenza cross-output.

    Premia quando:
    - Input A e B condividono un token (es: "Smith")
    - Output A' e B' condividono una VARIAZIONE di quel token (es: "Smithsonn")

    Returns:
        (bonus, details)
    """
    # Token condivisi negli input
    shared_input = _find_shared_tokens(input_A, input_B)
    if not shared_input:
        return 0.0, {"reason": "no_shared_input_tokens"}

    # Token condivisi negli output
    shared_output = _find_shared_tokens(output_A, output_B)
    if not shared_output:
        return 0.0, {"reason": "no_shared_output_tokens"}

    # Cerca variazioni creative: token output simili a token input
    creative_variations = []
    for out_tok in shared_output:
        for in_tok in shared_input:
            sim = _word_sim(out_tok, in_tok)
            # Variazione creativa: simile ma non identico (0.5 < sim < 0.95)
            if 0.5 < sim < 0.95:
                creative_variations.append({
                    "input_token": in_tok,
                    "output_token": out_tok,
                    "similarity": sim
                })

    if creative_variations:
        # Bonus proporzionale al numero di variazioni creative
        bonus = min(0.3, 0.15 * len(creative_variations))
        return bonus, {
            "reason": "coherent_creative_variation",
            "variations": creative_variations
        }

    # Se condividono token identici (non variati), piccolo bonus
    exact_shared = shared_input & shared_output
    if exact_shared:
        return 0.1, {
            "reason": "coherent_preserved_tokens",
            "tokens": list(exact_shared)
        }

    return 0.0, {"reason": "no_coherent_variation"}


def calculate_pair_score(input_A: str, input_B: str, output_A: str, output_B: str) -> Dict:
    """
    Calcola score per una COPPIA di output, includendo bonus di coerenza.
    Score normalizzato in [0, 1].

    Esempio eccellente:
        Input:  "Mary Smith" + "John Smith"
        Output: "M. Smithsonn" + "J. Smithsonn"

        - Score individuale alto (mix di elementi)
        - BONUS: "Smithsonn" è variazione coerente di "Smith" condiviso

    Returns:
        {
            "score": float,  # Score finale normalizzato [0, 1]
            "score_A": float,
            "score_B": float,
            "coherence_bonus": float,
            "details": {...}
        }
    """
    # Score individuali
    result_A = calculate_score_detailed(input_A, output_A, input_B)
    result_B = calculate_score_detailed(input_B, output_B, input_A)

    score_A = max(0, result_A["score"])  # Ignora -1 (empty/prompt_leak)
    score_B = max(0, result_B["score"])

    # Bonus coerenza cross-output
    coherence_bonus, coherence_details = _coherence_bonus(
        input_A, input_B, output_A, output_B
    )

    # Bonus parallel: entrambi gli output sono buone variazioni
    parallel_bonus = 0.0
    if score_A >= 0.6 and score_B >= 0.6:
        parallel_bonus = 0.1
        coherence_details["parallel_bonus"] = True

    # Score finale normalizzato [0, 1]
    # I bonus "riempiono" lo spazio rimanente fino a 1.0
    base_score = (score_A + score_B) / 2
    total_bonus = coherence_bonus + parallel_bonus
    final_score = min(1.0, base_score + total_bonus * (1.0 - base_score))

    return {
        "score": final_score,
        "base_score": base_score,
        "score_A": score_A,
        "score_B": score_B,
        "reason_A": result_A.get("reason"),
        "reason_B": result_B.get("reason"),
        "coherence_bonus": coherence_bonus,
        "parallel_bonus": parallel_bonus,
        "coherence_details": coherence_details,
    }


def calculate_score(orig: str, gen: str, other: str = None) -> float:
    """Calcola lo score (0-1) per un output generato."""
    result = calculate_score_detailed(orig, gen, other)
    return result["score"]


def calculate_score_detailed(orig: str, gen: str, other: str = None) -> Dict:
    """Calcola lo score con dettagli."""

    gen_c = gen.strip()
    orig_c = orig.strip()

    # Filtri base
    if len(gen_c) < 2:
        return {"score": -1.0, "reason": "empty"}

    if any(x in gen_c.lower() for x in ["rewrite", "paraphrase"]):
        return {"score": -1.0, "reason": "prompt_leak"}

    # Determina se testo corto o lungo
    word_count = len(re.findall(r'\w+', orig_c))

    if word_count <= SHORT_TEXT_THRESHOLD:
        return _score_short_text(orig, gen, other)
    else:
        return _score_long_text(orig, gen, other)


# === TEST ===
if __name__ == "__main__":
    print("=== TEST PAIR SCORING (COERENZA CROSS-OUTPUT) ===\n")

    pair_tests = [
        # (input_A, input_B, output_A, output_B, description)
        ("Mary Smith", "John Smith", "M. Smithsonn", "J. Smithsonn",
         "ECCELLENTE: variazione coerente di Smith"),
        ("Mary Smith", "John Smith", "M. Smith", "J. Smith",
         "BUONO: abbreviazione coerente, Smith preservato"),
        ("Mary Smith", "John Smith", "Maria", "Giovanni",
         "OK: traduzione ma perde coerenza"),
        ("Mary Smith", "John Smith", "xyz", "abc",
         "GARBAGE: nessuna connessione"),
        ("Alice Johnson", "Bob Williams", "A. Johnsonn", "B. Williamson",
         "BUONO: variazioni individuali ma non coerenti tra loro"),
        ("Dohn John", "Dohn Johnatan", "John Dohn", "Johnatan Dohn",
         "SHUFFLE: solo riordino token - penalizzato"),
        # Nuovi test da report
        ("rollins band", "rollins band", "rollins the band", "rollins the band",
         "FILLER: solo aggiunta 'the' - penalizzato"),
        ("supertramp", "supertramp", "supertramp the", "supertramp the",
         "FILLER: solo aggiunta 'the' - penalizzato"),
        ("hollywood actor", "clint clinton", "u00e9l b octavian", "u00e9l stefano",
         "UNICODE GARBAGE: caratteri corrotti"),
    ]

    for inA, inB, outA, outB, desc in pair_tests:
        result = calculate_pair_score(inA, inB, outA, outB)
        print(f"[{result['score']:.2f}] {desc}")
        print(f"    Input:  '{inA}' + '{inB}'")
        print(f"    Output: '{outA}' + '{outB}'")
        bonus_str = f"coh={result['coherence_bonus']:.2f} par={result['parallel_bonus']:.2f}"
        print(f"    Base: {result['base_score']:.2f} | Bonus: {bonus_str}")
        if result['coherence_details'].get('variations'):
            for v in result['coherence_details']['variations']:
                print(f"    → '{v['input_token']}' → '{v['output_token']}' (sim: {v['similarity']:.2f})")
        print()

    print("=== TEST SCORING - TESTI CORTI ===\n")

    short_tests = [
        ("John Smith", "Mary Johnson", "J. Marith", "perfect_mix"),
        ("John Smith", "Mary Johnson", "M. Jonathan", "perfect_mix"),
        ("John Smith", "Mary Johnson", "John Smith", "identity"),
        ("John Smith", "Mary Johnson", "Mary Johnson", "identity"),
        ("John Smith", "John Smith", "Jhon Smitthson", "good_variation_identical_inputs"),
        ("John Smith", "John Smith", "John Smith", "identity"),
        ("John Smith", "Mary Johnson", "Jonathan", "partial_connection"),
        ("John Smith", "Mary Johnson", "xyzabc", "garbage"),
        # === NUOVI TEST v10 ===
        # Solo jr (penalizzato)
        ("John Smith", "Mary Johnson", "John Smith jr", "only_lazy_suffix"),
        ("Scott Dominic", "Scott Dominic", "Scott Dominic jr", "only_lazy_suffix"),
        # jr CON altre modifiche (OK!)
        ("John Smith", "Mary Johnson", "Jhon Smith jr", "all_words_modified"),
        # Parole impronunciabili (penalizzato forte)
        ("John Smith", "Mary Johnson", "Jhn Smth", "unpronounceable"),
        ("Freddi Scott", "Freddi Scott", "Frd Sctt", "unpronounceable"),
        # Multi-word tutte modificate (bonus!)
        ("Scott Dominic", "Scott Dominic", "Scot Dominicson", "all_words_modified"),
        ("John William Smith", "John William Smith", "Jhon Willam Smyth", "all_words_modified"),
    ]

    for A, B, gen, expected in short_tests:
        result = calculate_score_detailed(A, gen, B)
        status = "✓" if result["reason"] == expected else "✗"
        print(f"{status} '{A}' + '{B}' -> '{gen}'")
        print(f"   Score: {result['score']:.1f} | {result['reason']} (exp: {expected})")
        print()

    print("\n=== TEST SCORING - TESTI LUNGHI ===\n")

    long_A = "The quick brown fox jumps over the lazy dog in the garden"
    long_B = "A fast red fox leaps across the sleeping hound near the house"

    long_tests = [
        # (A, B, gen, description)
        (long_A, long_B, long_A, "Identity (copia esatta)"),
        (long_A, long_B, "A swift brown fox leaps over the tired dog", "Riformulazione (sinonimi)"),
        (long_A, long_B, "Fox jumps over dog in garden", "Summarization"),
        (long_A, long_B, "The brown fox quickly jumped over the lazy canine", "Cambio stile"),
        (long_A, long_B, "xyz abc def ghi jkl mno pqr stu", "Garbage (no connessione)"),
    ]

    for A, B, gen, desc in long_tests:
        result = calculate_score_detailed(A, gen, B)
        print(f"[{result['score']:.1f}] {desc}")
        print(f"    Reason: {result['reason']}")
        if 'semantic_sim' in result:
            print(f"    Semantic: {result['semantic_sim']:.2f}")
        print()
