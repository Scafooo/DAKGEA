"""Literal encoder: word2vec → flat vectors → AutoEncoder → dim-d embeddings."""

from __future__ import annotations

import numpy as np
from sklearn import preprocessing

from .auto_encoder import train_autoencoder


TOKENS_MAX_LEN = 5
WORD2VEC_DIM = 300


def read_word2vec(file_path: str, vector_dimension: int = WORD2VEC_DIM) -> dict:
    word2vec = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split(' ')
            if len(parts) != vector_dimension + 1:
                continue
            word2vec[parts[0]] = np.array(list(map(float, parts[1:])), dtype=np.float32)
    return word2vec


def _generate_char_word2vec(word_list: list, vector_dimension: int = WORD2VEC_DIM) -> dict:
    """Fallback: character-level word2vec for OOV words via gensim."""
    try:
        from gensim.models.word2vec import Word2Vec
    except ImportError:
        return {}

    char_sequences = [list(w) for w in word_list]
    model = Word2Vec(char_sequences, vector_size=vector_dimension, window=5, min_count=1, workers=1)
    word2vec = {}
    for word in word_list:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        count = 0
        for ch in word:
            if ch in model.wv:
                vec += model.wv[ch]
                count += 1
        if count > 0:
            word2vec[word] = vec / count
    return word2vec


def enrich_word2vec(word2vec: dict, literal_list: list) -> dict:
    """Add character embeddings for OOV words."""
    oov = []
    for literal in literal_list:
        for word in literal.split(' '):
            if word and word not in word2vec:
                oov.append(word)
    if oov:
        char_vecs = _generate_char_word2vec(list(set(oov)))
        word2vec = {**word2vec, **char_vecs}
    return word2vec


def clear_attribute_triples(attribute_triples):
    """Filter and clean attribute triples (remove rare attrs, clean literal strings)."""
    attr_count = {}
    for _, a, _ in attribute_triples:
        attr_count[a] = attr_count.get(a, 0) + 1
    freq_attrs = {a for a, c in attr_count.items() if c >= 10}

    cleaned = []
    for e, a, v in attribute_triples:
        if a not in freq_attrs:
            continue
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        v = (v.replace('.', '').replace('(', '').replace(')', '')
              .replace(',', '').replace('"', '')
              .replace('_', ' ').replace('-', ' ').replace('/', ' '))
        if 'http' in v:
            continue
        cleaned.append((e, a, v))
    return cleaned


def encode_literals(literal_list: list, word2vec: dict, dim: int = 75,
                    hidden_dims=None, activation: str = 'tanh',
                    normalize: bool = True, epochs: int = 100,
                    batch_size: int = 512, learning_rate: float = 0.001,
                    optimizer_name: str = 'Adagrad') -> np.ndarray:
    """
    Encode a list of literal strings to dim-d vectors using word2vec + AutoEncoder.
    Returns numpy array of shape [len(literal_list), dim].
    """
    word2vec = enrich_word2vec(word2vec, literal_list)

    # Build word2vec flat vectors [n_literals, TOKENS_MAX_LEN * WORD2VEC_DIM]
    input_dim = TOKENS_MAX_LEN * WORD2VEC_DIM
    vecs = np.zeros((len(literal_list), input_dim), dtype=np.float32)
    for i, literal in enumerate(literal_list):
        words = literal.split(' ')
        for j in range(min(TOKENS_MAX_LEN, len(words))):
            if words[j] in word2vec:
                vecs[i, j * WORD2VEC_DIM:(j + 1) * WORD2VEC_DIM] = word2vec[words[j]]

    model = train_autoencoder(
        vecs, dim=dim, hidden_dims=hidden_dims, activation=activation,
        normalize=normalize, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, optimizer_name=optimizer_name)

    encoded = model.encode_batches(vecs, batch_size=batch_size)
    return encoded
