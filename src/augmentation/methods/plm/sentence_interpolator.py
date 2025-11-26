"""Sentence-level interpolation for long text attributes.

This module handles the interpolation of long text attributes (e.g., descriptions,
comments, biographies) by splitting them into sentences, chunking them to fit
within BART's token limits, and interpolating each chunk separately.
"""

import re
from typing import List, Tuple, TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex.

    Handles common abbreviations and punctuation patterns.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []

    # Preserve common abbreviations from splitting
    text = text.replace("Dr.", "Dr<DOT>")
    text = text.replace("Mr.", "Mr<DOT>")
    text = text.replace("Mrs.", "Mrs<DOT>")
    text = text.replace("Ms.", "Ms<DOT>")
    text = text.replace("Inc.", "Inc<DOT>")
    text = text.replace("Ltd.", "Ltd<DOT>")
    text = text.replace("Sr.", "Sr<DOT>")
    text = text.replace("Jr.", "Jr<DOT>")
    text = text.replace("etc.", "etc<DOT>")
    text = text.replace("e.g.", "e<DOT>g<DOT>")
    text = text.replace("i.e.", "i<DOT>e<DOT>")

    # Split on sentence boundaries: . ! ? followed by space and capital letter or end of string
    # Also split on newlines as they often indicate sentence boundaries
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$|\n+'
    sentences = re.split(pattern, text)

    # Restore abbreviations and clean
    sentences = [
        s.replace("<DOT>", ".").strip()
        for s in sentences
        if s and s.strip()
    ]

    return sentences


def count_tokens(text: str, tokenizer: "PreTrainedTokenizer") -> int:
    """Count tokens in text using the provided tokenizer.

    Args:
        text: Input text
        tokenizer: Tokenizer to use for counting

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def group_sentences_into_chunks(
    sentences: List[str],
    tokenizer: "PreTrainedTokenizer",
    max_tokens: int = 80
) -> List[List[str]]:
    """Group sentences into chunks that fit within max_tokens.

    Args:
        sentences: List of sentences to group
        tokenizer: Tokenizer to use for token counting
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunks, where each chunk is a list of sentences
    """
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sent_length = count_tokens(sentence, tokenizer)

        # If a single sentence exceeds max_tokens, put it alone
        # (it will be truncated during interpolation)
        if sent_length > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            chunks.append([sentence])
            logger.debug(f"[SENTENCE_CHUNKING] Single sentence exceeds max_tokens ({sent_length} > {max_tokens}), creating solo chunk")
            continue

        # If adding this sentence would exceed limit, finalize current chunk
        if current_length + sent_length > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_length = sent_length
        else:
            current_chunk.append(sentence)
            current_length += sent_length

    # Add remaining sentences
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def interpolate_long_text(
    text1: str,
    text2: str,
    interpolate_fn,
    tokenizer: "PreTrainedTokenizer",
    max_tokens: int = 80,
    min_length_for_chunking: int = 60,
) -> Tuple[str, str]:
    """Interpolate long text by splitting into sentence chunks.

    Args:
        text1: First text (source)
        text2: Second text (target)
        interpolate_fn: Function to interpolate a pair of texts
                       Should have signature: (str, str) -> Tuple[str, str]
        tokenizer: Tokenizer for token counting
        max_tokens: Maximum tokens per chunk
        min_length_for_chunking: Minimum text length (in tokens) to trigger chunking

    Returns:
        Tuple of (interpolated_text1, interpolated_text2)
    """
    # Check if texts are long enough to need chunking
    len1 = count_tokens(text1, tokenizer)
    len2 = count_tokens(text2, tokenizer)

    max_len = max(len1, len2)

    if max_len < min_length_for_chunking:
        # Texts are short enough, use standard interpolation
        logger.debug(f"[SENTENCE_INTERPOLATION] Texts short enough ({max_len} < {min_length_for_chunking}), using standard interpolation")
        return interpolate_fn(text1, text2)

    logger.info(f"[SENTENCE_INTERPOLATION] Long text detected ({max_len} tokens), using sentence-level interpolation")

    # Split into sentences
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)

    logger.debug(f"[SENTENCE_INTERPOLATION] Split into {len(sentences1)} and {len(sentences2)} sentences")

    # Group into chunks
    chunks1 = group_sentences_into_chunks(sentences1, tokenizer, max_tokens)
    chunks2 = group_sentences_into_chunks(sentences2, tokenizer, max_tokens)

    logger.debug(f"[SENTENCE_INTERPOLATION] Created {len(chunks1)} and {len(chunks2)} chunks")

    # Align chunks: use min length to avoid index errors
    n_chunks = min(len(chunks1), len(chunks2))

    if len(chunks1) != len(chunks2):
        logger.warning(
            f"[SENTENCE_INTERPOLATION] Chunk count mismatch: {len(chunks1)} vs {len(chunks2)}, "
            f"will use first {n_chunks} chunks"
        )

    # Interpolate each chunk pair
    interpolated_chunks1 = []
    interpolated_chunks2 = []

    for i in range(n_chunks):
        chunk_text1 = " ".join(chunks1[i])
        chunk_text2 = " ".join(chunks2[i])

        logger.debug(f"[SENTENCE_INTERPOLATION] Interpolating chunk {i+1}/{n_chunks}")
        logger.debug(f"  Chunk 1 ({count_tokens(chunk_text1, tokenizer)} tokens): {chunk_text1[:100]}...")
        logger.debug(f"  Chunk 2 ({count_tokens(chunk_text2, tokenizer)} tokens): {chunk_text2[:100]}...")

        # Interpolate this chunk pair
        interp1, interp2 = interpolate_fn(chunk_text1, chunk_text2)

        interpolated_chunks1.append(interp1)
        interpolated_chunks2.append(interp2)

    # Handle remaining chunks if one text has more chunks
    if len(chunks1) > n_chunks:
        logger.debug(f"[SENTENCE_INTERPOLATION] Appending {len(chunks1) - n_chunks} remaining chunks from text1")
        for i in range(n_chunks, len(chunks1)):
            chunk_text = " ".join(chunks1[i])
            interpolated_chunks1.append(chunk_text)

    if len(chunks2) > n_chunks:
        logger.debug(f"[SENTENCE_INTERPOLATION] Appending {len(chunks2) - n_chunks} remaining chunks from text2")
        for i in range(n_chunks, len(chunks2)):
            chunk_text = " ".join(chunks2[i])
            interpolated_chunks2.append(chunk_text)

    # Rejoin chunks with proper spacing
    result1 = " ".join(interpolated_chunks1)
    result2 = " ".join(interpolated_chunks2)

    logger.info(f"[SENTENCE_INTERPOLATION] Completed: {len(interpolated_chunks1)} chunks → {count_tokens(result1, tokenizer)} tokens")

    return result1, result2


def is_long_text_predicate(predicate: str) -> bool:
    """Check if a predicate typically contains long text.

    Args:
        predicate: Predicate URI or name

    Returns:
        True if predicate typically contains long text
    """
    predicate_lower = predicate.lower()

    # Common long-text predicates
    long_text_indicators = [
        'comment',
        'description',
        'bio',
        'biography',
        'summary',
        'abstract',
        'note',
        'text',
        'content',
        'article',
        'review',
    ]

    return any(indicator in predicate_lower for indicator in long_text_indicators)
