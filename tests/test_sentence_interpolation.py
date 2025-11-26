"""Test sentence-level interpolation for long text attributes."""

from transformers import BartTokenizer

from src.augmentation.methods.plm.sentence_interpolator import (
    split_into_sentences,
    count_tokens,
    group_sentences_into_chunks,
    interpolate_long_text,
    is_long_text_predicate,
)


def test_split_into_sentences():
    """Test sentence splitting with various punctuation patterns."""
    # Basic splitting
    text = "This is sentence one. This is sentence two! Is this sentence three?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "This is sentence one."
    assert sentences[1] == "This is sentence two!"
    assert sentences[2] == "Is this sentence three?"

    # Handle abbreviations
    text = "Dr. Smith works at Inc. Corp."
    sentences = split_into_sentences(text)
    # Should NOT split on "Dr." or "Inc."
    assert len(sentences) == 1

    # Multiple sentences with abbreviations
    text = "Dr. John works here. He is a Sr. engineer."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2

    # Newlines as sentence boundaries
    text = "First line.\nSecond line.\nThird line."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3


def test_count_tokens():
    """Test token counting."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    text = "This is a test sentence."
    count = count_tokens(text, tokenizer)
    assert count > 0
    assert isinstance(count, int)

    # Empty text
    assert count_tokens("", tokenizer) == 0
    assert count_tokens(None, tokenizer) == 0


def test_group_sentences_into_chunks():
    """Test chunking sentences by token count."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    sentences = [
        "This is a short sentence.",
        "This is another short sentence.",
        "And one more.",
    ]

    # With large max_tokens, all should fit in one chunk
    chunks = group_sentences_into_chunks(sentences, tokenizer, max_tokens=100)
    assert len(chunks) == 1
    assert len(chunks[0]) == 3

    # With small max_tokens, should split into multiple chunks
    chunks = group_sentences_into_chunks(sentences, tokenizer, max_tokens=10)
    assert len(chunks) >= 2  # At least 2 chunks


def test_group_sentences_with_long_sentence():
    """Test chunking when a single sentence exceeds max_tokens."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    sentences = [
        "Short.",
        "This is an extremely long sentence that contains many many words and should definitely exceed the maximum token limit that we set for chunking purposes in this specific test case.",
        "Another short.",
    ]

    chunks = group_sentences_into_chunks(sentences, tokenizer, max_tokens=20)

    # The long sentence should be in its own chunk
    assert len(chunks) >= 2
    # Find the chunk with the long sentence
    long_chunk = None
    for chunk in chunks:
        if any("extremely long sentence" in s for s in chunk):
            long_chunk = chunk
            break
    assert long_chunk is not None
    assert len(long_chunk) == 1  # Should be alone


def test_interpolate_long_text_short_input():
    """Test that short texts bypass sentence-level interpolation."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    text1 = "Short text."
    text2 = "Brief text."

    # Mock interpolation function that just concatenates
    def mock_interpolate(t1, t2):
        return f"INTERP:{t1}", f"INTERP:{t2}"

    result1, result2 = interpolate_long_text(
        text1, text2,
        mock_interpolate,
        tokenizer,
        max_tokens=80,
        min_length_for_chunking=60
    )

    # Should use standard interpolation (short text)
    assert result1 == "INTERP:Short text."
    assert result2 == "INTERP:Brief text."


def test_interpolate_long_text_long_input():
    """Test sentence-level interpolation with long text."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # Create long texts (>60 tokens)
    text1 = " ".join([f"Sentence {i} in text one." for i in range(20)])
    text2 = " ".join([f"Sentence {i} in text two." for i in range(20)])

    # Mock interpolation function
    call_count = {"count": 0}

    def mock_interpolate(t1, t2):
        call_count["count"] += 1
        return f"CHUNK{call_count['count']}:{t1[:20]}...", f"CHUNK{call_count['count']}:{t2[:20]}..."

    result1, result2 = interpolate_long_text(
        text1, text2,
        mock_interpolate,
        tokenizer,
        max_tokens=30,  # Small chunks
        min_length_for_chunking=50
    )

    # Should have called interpolation multiple times (chunking occurred)
    assert call_count["count"] > 1

    # Results should be concatenated chunks
    assert "CHUNK1:" in result1
    assert "CHUNK1:" in result2


def test_is_long_text_predicate():
    """Test identification of long-text predicates."""
    # Common long-text predicates
    assert is_long_text_predicate("rdfs:comment")
    assert is_long_text_predicate("http://example.org/description")
    assert is_long_text_predicate("bio")
    assert is_long_text_predicate("biography")
    assert is_long_text_predicate("summary")
    assert is_long_text_predicate("abstract")
    assert is_long_text_predicate("review")

    # Non-long-text predicates
    assert not is_long_text_predicate("name")
    assert not is_long_text_predicate("age")
    assert not is_long_text_predicate("title")
    assert not is_long_text_predicate("rdfs:label")


def test_sentence_chunking_preserves_order():
    """Test that sentence chunking preserves sentence order."""
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    sentences = [f"Sentence {i}." for i in range(10)]

    chunks = group_sentences_into_chunks(sentences, tokenizer, max_tokens=20)

    # Flatten chunks and verify order
    flattened = []
    for chunk in chunks:
        flattened.extend(chunk)

    assert flattened == sentences  # Order preserved


if __name__ == "__main__":
    # Run basic tests
    print("Testing split_into_sentences...")
    test_split_into_sentences()
    print("✓ Passed")

    print("\nTesting count_tokens...")
    test_count_tokens()
    print("✓ Passed")

    print("\nTesting group_sentences_into_chunks...")
    test_group_sentences_into_chunks()
    print("✓ Passed")

    print("\nTesting group_sentences_with_long_sentence...")
    test_group_sentences_with_long_sentence()
    print("✓ Passed")

    print("\nTesting interpolate_long_text_short_input...")
    test_interpolate_long_text_short_input()
    print("✓ Passed")

    print("\nTesting interpolate_long_text_long_input...")
    test_interpolate_long_text_long_input()
    print("✓ Passed")

    print("\nTesting is_long_text_predicate...")
    test_is_long_text_predicate()
    print("✓ Passed")

    print("\nTesting sentence_chunking_preserves_order...")
    test_sentence_chunking_preserves_order()
    print("✓ Passed")

    print("\n✅ All tests passed!")
