"""
Easy Data Augmentation techniques for text classification.
Original: Jason Wei and Kai Zou (https://github.com/jasonwei20/eda_nlp)
Adapted for DAKGEA: Python-3-clean, no global seed, no CLI dependency.
"""

from __future__ import annotations

import random
import re
from random import shuffle

from nltk.corpus import wordnet

_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "",
}

_ALPHA_CHARS = set("qwertyuiopasdfghjklzxcvbnm ")


def _clean(line: str) -> str:
    line = line.replace("'", "").replace("'", "")
    line = line.replace("-", " ").replace("\t", " ").replace("\n", " ")
    line = line.lower()
    line = "".join(c if c in _ALPHA_CHARS else " " for c in line)
    line = re.sub(r" +", " ", line).strip()
    return line


def _get_synonyms(word: str) -> list[str]:
    synonyms: set[str] = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            s = lemma.name().replace("_", " ").replace("-", " ").lower()
            s = "".join(c for c in s if c in _ALPHA_CHARS)
            synonyms.add(s)
    synonyms.discard(word)
    return list(synonyms)


def _synonym_replacement(words: list[str], n: int) -> list[str]:
    new_words = words.copy()
    candidates = [w for w in words if w not in _STOP_WORDS]
    random.shuffle(candidates)
    replaced = 0
    for word in candidates:
        syns = _get_synonyms(word)
        if syns:
            new_words = [random.choice(syns) if w == word else w for w in new_words]
            replaced += 1
        if replaced >= n:
            break
    return " ".join(new_words).split()


def _random_insertion(words: list[str], n: int) -> list[str]:
    new_words = words.copy()
    for _ in range(n):
        syns: list[str] = []
        attempts = 0
        while not syns and attempts < 10:
            syns = _get_synonyms(random.choice(new_words))
            attempts += 1
        if syns:
            new_words.insert(random.randint(0, len(new_words) - 1), syns[0])
    return new_words


def _random_swap(words: list[str], n: int) -> list[str]:
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            break
        i = random.randint(0, len(new_words) - 1)
        j = i
        attempts = 0
        while j == i and attempts < 4:
            j = random.randint(0, len(new_words) - 1)
            attempts += 1
        if j != i:
            new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words


def _random_deletion(words: list[str], p: float) -> list[str]:
    if len(words) == 1:
        return words
    result = [w for w in words if random.random() > p]
    return result if result else [random.choice(words)]


def eda(
    sentence: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 9,
) -> list[str]:
    """
    Return a list of augmented variants of *sentence* (original appended last).

    Parameters
    ----------
    alpha_sr : fraction of words to replace with synonyms
    alpha_ri : fraction of words to insert as synonyms
    alpha_rs : fraction of words to randomly swap
    p_rd     : per-word deletion probability
    num_aug  : desired number of augmented sentences (excluding original)
    """
    sentence = _clean(sentence)
    words = [w for w in sentence.split(" ") if w]
    if not words:
        return [sentence]

    n_words = len(words)
    per_technique = int(num_aug / 4) + 1
    augmented: list[str] = []

    if alpha_sr > 0:
        n = max(1, int(alpha_sr * n_words))
        for _ in range(per_technique):
            augmented.append(" ".join(_synonym_replacement(words, n)))

    if alpha_ri > 0:
        n = max(1, int(alpha_ri * n_words))
        for _ in range(per_technique):
            augmented.append(" ".join(_random_insertion(words, n)))

    if alpha_rs > 0:
        n = max(1, int(alpha_rs * n_words))
        for _ in range(per_technique):
            augmented.append(" ".join(_random_swap(words, n)))

    if p_rd > 0:
        for _ in range(per_technique):
            augmented.append(" ".join(_random_deletion(words, p_rd)))

    augmented = [_clean(s) for s in augmented]
    shuffle(augmented)

    if num_aug >= 1:
        augmented = augmented[:num_aug]
    else:
        keep = num_aug / max(len(augmented), 1)
        augmented = [s for s in augmented if random.random() < keep]

    augmented.append(sentence)
    return augmented
