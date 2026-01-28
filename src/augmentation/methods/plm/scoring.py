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


def _is_abbreviation(short: str, full: str) -> bool:
    """Verifica se 'short' è un'abbreviazione di 'full'. Es: J -> John"""
    if len(short) == 1:
        return full.lower().startswith(short.lower())
    if len(short) <= 3 and len(full) > 3:
        return full.lower().startswith(short.lower())
    return False


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

    # Identity check
    if gen_c == orig_c or gen_c == other_c:
        return {"score": 0.0, "reason": "identity"}

    # Near-copy check
    sim_A = _word_sim(gen_c, orig_c)
    sim_B = _word_sim(gen_c, other_c) if other_c else 0.0

    if sim_A > 0.9 or sim_B > 0.9:
        return {"score": 0.2, "reason": "near_copy", "sim": max(sim_A, sim_B)}

    # Analisi parole
    gen_words = set(re.findall(r'\w+', gen_c))
    words_A = set(re.findall(r'\w+', orig_c))
    words_B = set(re.findall(r'\w+', other_c)) if other_c else set()

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
            "score": 0.6,
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

    Esempio eccellente:
        Input:  "Mary Smith" + "John Smith"
        Output: "M. Smithsonn" + "J. Smithsonn"

        - Score individuale alto (mix di elementi)
        - BONUS: "Smithsonn" è variazione coerente di "Smith" condiviso

    Returns:
        {
            "score": float,  # Score finale (0-1.3 con bonus)
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

    # Score finale
    base_score = (score_A + score_B) / 2
    final_score = min(1.3, base_score + coherence_bonus)  # Cap a 1.3

    return {
        "score": final_score,
        "base_score": base_score,
        "score_A": score_A,
        "score_B": score_B,
        "reason_A": result_A.get("reason"),
        "reason_B": result_B.get("reason"),
        "coherence_bonus": coherence_bonus,
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
    ]

    for inA, inB, outA, outB, desc in pair_tests:
        result = calculate_pair_score(inA, inB, outA, outB)
        print(f"[{result['score']:.2f}] {desc}")
        print(f"    Input:  '{inA}' + '{inB}'")
        print(f"    Output: '{outA}' + '{outB}'")
        print(f"    Base: {result['base_score']:.2f} | Bonus: +{result['coherence_bonus']:.2f}")
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
