"""Training data builder for Mix-up T5/BART with Creative Variations.

Includes all optimizations from run_xl_pipeline.py:
- Aligned pairs (3x weight, bidirectional)
- Learned variations from aligned pairs
- Creative synthetic variations
- Orphan variations
- Flip pairs (yes/no, true/false, etc.)
- Multi-word algorithmic variations
- Quality filtering (unicode, swaps, filler, similarity)
"""

import random
import logging
import re
import codecs
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher
from pathlib import Path

from rdflib import Literal
from src.core.dataset import Dataset

logger = logging.getLogger(__name__)


# ============================================================
# FLIP PAIRS for training
# ============================================================
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

FILLER_WORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for'}

UNICODE_GARBAGE_PATTERNS = [
    re.compile(r'u00[a-f0-9]{2}', re.IGNORECASE),
    re.compile(r'\\u00[a-f0-9]{2}', re.IGNORECASE),
    re.compile(r'&#x[a-f0-9]{2,4};', re.IGNORECASE),
    re.compile(r'&#\d{2,4};'),
    re.compile(r'%[a-f0-9]{2}', re.IGNORECASE),
]


# ============================================================
# Helper Functions
# ============================================================
def _local_name(uri: str) -> str:
    return str(uri).split("/")[-1].split("#")[-1].upper()


def load_attr_names(dataset_path: str) -> dict:
    """Load attribute name mappings from dataset."""
    from pathlib import Path
    attr_map = {}
    dataset_dir = Path(dataset_path)
    for i in [1, 2]:
        path = dataset_dir / f"attribute_data/attr_names{i}"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        attr_map[parts[0].strip()] = parts[1].strip()
    return attr_map


def clean_predicate(uri: str, attr_map: dict) -> str:
    """Get clean predicate name using attr_map or falling back to URI extraction."""
    uri_str = str(uri)
    if uri_str in attr_map:
        return attr_map[uri_str].replace(' ', '_').lower()
    return uri_str.split('/')[-1].split('#')[-1].replace('>', '').replace('<', '').lower()


def fix_unicode_escapes(text: str) -> str:
    """Decode common unicode escapes."""
    result = text

    def replace_u00(match):
        try:
            hex_val = match.group(0)[1:]
            return chr(int(hex_val, 16))
        except:
            return match.group(0)

    result = re.sub(r'u00[a-f0-9]{2}', replace_u00, result, flags=re.IGNORECASE)

    try:
        result = codecs.decode(result, 'unicode_escape')
    except:
        pass

    return result


def has_unicode_garbage(text: str) -> bool:
    return any(p.search(text) for p in UNICODE_GARBAGE_PATTERNS)


def has_vowel(word: str) -> bool:
    return any(c.lower() in 'aeiou' for c in word)


def min_edit_distance(s1: str, s2: str) -> int:
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
    if ": " in prompt:
        return prompt.split(": ", 1)[1].strip()
    return prompt.strip()


def is_only_filler_difference(input_text: str, target_text: str) -> bool:
    if ": " in input_text:
        input_val = input_text.split(": ", 1)[1].strip().lower()
    else:
        input_val = input_text.strip().lower()

    target_val = target_text.strip().lower()

    if input_val == target_val:
        return False

    input_tokens = input_val.split()
    target_tokens = target_val.split()

    input_no_filler = [t for t in input_tokens if t not in FILLER_WORDS]
    target_no_filler = [t for t in target_tokens if t not in FILLER_WORDS]

    return input_no_filler == target_no_filler


def is_token_swap(input_text: str, target_text: str) -> bool:
    if ": " in input_text:
        input_val = input_text.split(": ", 1)[1].strip()
    else:
        input_val = input_text.strip()

    target_val = target_text.strip()

    input_tokens = set(input_val.lower().split())
    target_tokens = set(target_val.lower().split())

    if input_tokens == target_tokens and input_val.lower() != target_val.lower():
        return True
    return False


# ============================================================
# Variation Generation
# ============================================================
def vary_word_algorithmic(word: str) -> str:
    """Vary a word using scalable algorithms (no dictionaries)."""
    if len(word) < 2:
        return word

    technique = random.choice([
        'suffix', 'suffix', 'suffix',
        'prefix',
        'truncate', 'truncate',
        'initial',
        'vowel_double',
        'vowel_swap'
    ])

    result = word

    if technique == 'suffix':
        suffix = random.choice(['y', 'ie', 'son', 'sen', 'man', 'er', 'ini', 'elli', 'ski', 'ov'])
        if word.endswith(suffix[0]):
            result = word + suffix[1:] if len(suffix) > 1 else word + suffix
        else:
            result = word + suffix

    elif technique == 'prefix':
        prefix = random.choice(['jr', 'mr', 'dr', 'st', 'von', 'de', 'van'])
        result = prefix + word

    elif technique == 'truncate' and len(word) >= 4:
        for length in [4, 3]:
            if len(word) > length:
                candidate = word[:length]
                if has_vowel(candidate):
                    result = candidate
                    break
        if result == word:
            for cut in range(1, min(3, len(word) - 2)):
                candidate = word[:-cut]
                if has_vowel(candidate) and len(candidate) >= 2:
                    result = candidate
                    break

    elif technique == 'initial':
        if len(word) > 3:
            result = word[0].upper() + '.'

    elif technique == 'vowel_double':
        vowels = 'aeiou'
        vowel_positions = [i for i, c in enumerate(word.lower()) if c in vowels]
        if vowel_positions:
            idx = random.choice(vowel_positions)
            result = word[:idx] + word[idx] + word[idx:]

    elif technique == 'vowel_swap':
        vowels = 'aeiou'
        vowel_positions = [i for i, c in enumerate(word.lower()) if c in vowels]
        if vowel_positions:
            idx = random.choice(vowel_positions)
            old_vowel = word[idx].lower()
            new_vowel = random.choice([v for v in vowels if v != old_vowel])
            result = word[:idx] + new_vowel + word[idx+1:]

    if not has_vowel(result):
        result = word + random.choice(['y', 'a', 'o'])

    return result


def learn_variations_from_pairs(aligned_pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Learn word variations from aligned pairs."""
    variations = {}

    for val_src, val_tgt in aligned_pairs:
        words_src = val_src.lower().split()
        words_tgt = val_tgt.lower().split()

        if len(words_src) != len(words_tgt):
            continue

        for ws, wt in zip(words_src, words_tgt):
            if ws == wt:
                continue

            sim = SequenceMatcher(None, ws, wt).ratio()
            if sim < 0.3 or sim > 0.95:
                continue

            if ws not in variations:
                variations[ws] = set()
            if wt not in variations:
                variations[wt] = set()

            variations[ws].add(wt)
            variations[wt].add(ws)

    return {k: list(v) for k, v in variations.items() if v}


def vary_word_with_learned(word: str, learned: Dict[str, List[str]]) -> str:
    """Vary a word using learned variations first, then algorithms."""
    word_lower = word.lower()

    if word_lower in learned and learned[word_lower]:
        if random.random() < 0.7:
            variant = random.choice(learned[word_lower])
            if word[0].isupper():
                variant = variant.capitalize()
            return variant

    return vary_word_algorithmic(word)


def vary_all_words(text: str, learned: Dict[str, List[str]] = None) -> str:
    """Vary every word in text."""
    if learned is None:
        learned = {}

    words = text.split()
    if len(words) < 2:
        return vary_word_with_learned(text, learned) if learned else vary_word_algorithmic(text)

    if learned:
        varied_words = [vary_word_with_learned(w, learned) for w in words]
    else:
        varied_words = [vary_word_algorithmic(w) for w in words]
    return ' '.join(varied_words)


# Import the full CreativeVariationGenerator
from src.augmentation.methods.plm.creative_variation_generator import CreativeVariationGenerator


# ============================================================
# Main Builder Class
# ============================================================
class MixupDataBuilder:
    """Build training data for Mix-up T5/BART with creative variations."""

    def __init__(self, confidence_threshold: float = 0.6, value_match_threshold: float = 0.3):
        self.value_match_threshold = value_match_threshold
        self.creative_gen = CreativeVariationGenerator()

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _generate_flip_pairs(self) -> List[Dict[str, str]]:
        """Generate flip training pairs (yes/no, true/false, etc.)."""
        pairs = []
        for val, flipped in FLIP_PAIRS.items():
            for _ in range(3):
                pairs.append({"input": f"generate variation <value>: {val}", "target": flipped})
            for _ in range(3):
                pairs.append({"input": f"generate variation <value>: {val}", "target": val})
        return pairs

    def _generate_multi_word_pairs(self, names: List[str], learned: Dict, n_per_name: int = 10) -> List[Dict[str, str]]:
        """Generate multi-word variation pairs."""
        pairs = []
        for name in names:
            words = name.split()
            if len(words) < 2:
                continue

            for _ in range(n_per_name):
                varied = vary_all_words(name, learned)
                if varied.lower() != name.lower():
                    pairs.append({
                        "input": f"generate variation <name>: {name}",
                        "target": varied
                    })

        return pairs

    def _filter_training_data(self, rows: List[Dict], min_edit_ratio: float = 0.1) -> List[Dict]:
        """Filter rows with unicode garbage, token swap, filler difference, or too similar."""
        filtered = []
        unicode_removed = 0
        swap_removed = 0
        filler_removed = 0
        too_similar_removed = 0

        for row in rows:
            inp, tgt = row['input'], row['target']

            tgt_fixed = fix_unicode_escapes(tgt)
            if tgt_fixed != tgt:
                tgt = tgt_fixed
                row['target'] = tgt_fixed

            inp_val = extract_value_from_prompt(inp)

            if has_unicode_garbage(inp) or has_unicode_garbage(tgt):
                unicode_removed += 1
                continue

            if is_token_swap(inp, tgt):
                swap_removed += 1
                continue

            if is_only_filler_difference(inp, tgt):
                filler_removed += 1
                continue

            max_len = max(len(inp_val), len(tgt))
            if max_len > 0:
                edit_dist = min_edit_distance(inp_val.lower(), tgt.lower())
                edit_ratio = edit_dist / max_len
                if edit_ratio < min_edit_ratio and inp_val.lower() != tgt.lower():
                    too_similar_removed += 1
                    continue

            filtered.append(row)

        logger.info(f"[FILTER] Removed: {unicode_removed} unicode, {swap_removed} swaps, {filler_removed} filler, {too_similar_removed} similar")
        logger.info(f"[FILTER] Kept: {len(filtered)}/{len(rows)} ({100*len(filtered)/len(rows) if rows else 0:.1f}%)")
        return filtered

    def build_training_data(self, dataset: Dataset, max_pairs_per_pred: int = 5000, dataset_path: str = None) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Build comprehensive training data with all optimizations.

        Args:
            dataset: The dataset object
            max_pairs_per_pred: Maximum pairs per predicate
            dataset_path: Optional path to dataset for loading attr_names

        Returns:
            Tuple of (training_rows, canonical_mapping)
        """
        logger.info("[MixupBuilder] Building Creative Variation training data...")

        # Load attribute name mappings if dataset_path provided
        attr_map = {}
        if dataset_path:
            attr_map = load_attr_names(dataset_path)
            logger.info(f"[MixupBuilder] Loaded {len(attr_map)} attribute name mappings")

        # Build canonical map
        canonical_map = {
            str(p): f"<{_local_name(p)}>"
            for p in (set(dataset.knowledge_graph_source.predicates()) |
                     set(dataset.knowledge_graph_target.predicates()))
        }

        # Store attr_map for use in loops
        self._attr_map = attr_map

        kg_src = dataset.knowledge_graph_source
        kg_tgt = dataset.knowledge_graph_target

        # Collect literals
        src_lits = defaultdict(list)
        for s, p, o in kg_src.triples((None, None, None)):
            if isinstance(o, Literal):
                src_lits[s].append((p, str(o)))

        tgt_lits = defaultdict(list)
        for s, p, o in kg_tgt.triples((None, None, None)):
            if isinstance(o, Literal):
                tgt_lits[s].append((p, str(o)))

        rows = []
        aligned_pairs = []
        real_pairs_count = 0
        synthetic_pairs_count = 0
        pred_counts = defaultdict(int)

        # A. ALIGNED PAIRS (3x weight, bidirectional)
        for s_uri, t_uri in dataset.aligned_entities:
            s_attrs = src_lits.get(s_uri, [])
            t_attrs = tgt_lits.get(t_uri, [])

            for ps, vs in s_attrs:
                p_name = clean_predicate(ps, self._attr_map).replace('_', ' ')
                p_tok = canonical_map.get(str(ps), f"<{p_name}>")

                if pred_counts[p_tok] >= max_pairs_per_pred:
                    continue

                for pt, vt in t_attrs:
                    if canonical_map.get(str(ps)) == canonical_map.get(str(pt)):
                        v1_c, v2_c = vs.strip().lower(), vt.strip().lower()

                        if v1_c != v2_c:
                            # BIDIRECTIONAL: real pairs duplicated 3x for weight
                            for _ in range(3):
                                rows.append({"input": f"generate variation <{p_name}>: {vs}", "target": vt})
                                rows.append({"input": f"generate variation <{p_name}>: {vt}", "target": vs})
                            real_pairs_count += 2
                            pred_counts[p_tok] += 1

                            # EXTRA: Synthetic variations (30% probability)
                            if random.random() < 0.3:
                                var_vs = self.creative_gen.generate(vs, vt, predicate=p_name)
                                if var_vs != vs and var_vs != vt:
                                    rows.append({"input": f"generate variation <{p_name}>: {vs}", "target": var_vs})
                                    synthetic_pairs_count += 1

                            if random.random() < 0.3:
                                var_vt = self.creative_gen.generate(vt, vs, predicate=p_name)
                                if var_vt != vt and var_vt != vs:
                                    rows.append({"input": f"generate variation <{p_name}>: {vt}", "target": var_vt})
                                    synthetic_pairs_count += 1

                        aligned_pairs.append((vs, vt))
                        break

        logger.info(f"[ALIGNED] Real pairs: {real_pairs_count} (x3 weight), Synthetic: {synthetic_pairs_count}")

        # Learn variations from aligned pairs
        learned_variations = learn_variations_from_pairs(aligned_pairs)
        logger.info(f"[LEARNED] Extracted {len(learned_variations)} word variations from aligned pairs")

        # B. ORPHANS with creative variations
        orphans_by_pred = defaultdict(list)
        for s, p, o in list(kg_src.triples((None, None, None))) + list(kg_tgt.triples((None, None, None))):
            if isinstance(o, Literal):
                val = str(o).strip()
                if val:
                    p_name = clean_predicate(p, self._attr_map).replace('_', ' ')
                    orphans_by_pred[p_name].append(val)

        orphan_count = 0
        for p_name, vals in orphans_by_pred.items():
            unique_vals = list(set(vals))
            selected = random.sample(unique_vals, min(len(unique_vals), 100))
            for v in selected:
                if random.random() < 0.5:
                    v_creative = self.creative_gen.generate(v, predicate=p_name)
                    if v_creative != v and len(v_creative) > 2:
                        rows.append({"input": f"generate variation <{p_name}>: {v}", "target": v_creative})
                        orphan_count += 1

        logger.info(f"[ORPHANS] Synthetic variations: {orphan_count}")

        # Filter quality
        logger.info(f"Pre-filter samples: {len(rows)}")
        rows = self._filter_training_data(rows)

        # C. FLIP PAIRS
        flip_pairs = self._generate_flip_pairs()
        rows.extend(flip_pairs)
        logger.info(f"[FLIP] Added {len(flip_pairs)} flip training pairs")

        # D. MULTI-WORD VARIATIONS
        multi_word_names = set()
        for row in rows:
            inp = row['input']
            if '<name>' in inp.lower():
                val = extract_value_from_prompt(inp)
                if len(val.split()) >= 2 and len(val) < 50:
                    multi_word_names.add(val)

        multi_word_pairs = self._generate_multi_word_pairs(list(multi_word_names), learned_variations, n_per_name=10)
        rows.extend(multi_word_pairs)
        logger.info(f"[MULTI-WORD] Added {len(multi_word_pairs)} pairs from {len(multi_word_names)} names")

        # Shuffle
        random.shuffle(rows)
        logger.info(f"Total training samples: {len(rows)}")

        return rows, canonical_map
