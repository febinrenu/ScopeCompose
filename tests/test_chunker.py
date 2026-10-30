"""The shared chunker: sentences are never split, remainders are merged, gold sentences are found."""

from __future__ import annotations

import pytest

from benchmark.chunker import Chunk, chunk_text, gold_chunk, split_sentences


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def sent(n, tag):
    """A sentence of n tokens. It starts with a capital, as real sentences do: the splitter only breaks
    after a full stop that is followed by a capital letter or a digit."""
    w = words(n, tag)
    return w[0].upper() + w[1:] + "."


def test_sentences_follow_punctuation_and_the_one_block_per_line_layout():
    text = "First rule applies. Second one follows.\nA new block starts here. Done!"
    assert split_sentences(text) == ["First rule applies.", "Second one follows.", "A new block starts here.", "Done!"]


def test_a_decimal_or_abbreviation_style_figure_does_not_split_a_sentence():
    assert split_sentences("The fee is 2.50 per withdrawal.") == ["The fee is 2.50 per withdrawal."]


def test_a_sentence_is_never_split_and_chunks_respect_the_target():
    text = " ".join(sent(30, f"s{k}x") for k in range(10))
    chunks = chunk_text("d", text, target_words=70, min_words=0)
    assert all(c.text.count(".") * 30 == c.n_words for c in chunks), "whole sentences only"
    assert all(c.n_words <= 70 for c in chunks)
    assert " ".join(c.text for c in chunks) == text


def test_an_overlong_sentence_becomes_its_own_chunk():
    text = sent(200, "long") + " " + sent(10, "short")
    chunks = chunk_text("d", text, target_words=50, min_words=0)
    assert chunks[0].n_words == 200 and chunks[1].n_words == 10


def test_a_short_remainder_is_merged_into_the_previous_chunk():
    text = sent(60, "a") + " " + sent(60, "b") + " " + sent(5, "c")
    merged = chunk_text("d", text, target_words=60, min_words=20)
    assert len(merged) == 2 and merged[-1].text.endswith(sent(5, "c"))
    unmerged = chunk_text("d", text, target_words=60, min_words=0)
    assert len(unmerged) == 3


def test_exactly_two_groups_with_a_short_tail_merge_without_error():
    """Regression: `groups[-2] += groups.pop()` indexed after the pop and raised IndexError here."""
    chunks = chunk_text("d", sent(60, "a") + " " + sent(5, "b"), target_words=60, min_words=20)
    assert len(chunks) == 1 and chunks[0].n_words == 65


def test_empty_and_whitespace_documents_have_no_chunks():
    assert chunk_text("d", "") == [] and chunk_text("d", "  \n \n") == []


def test_ids_are_stable_and_ordered():
    chunks = chunk_text("doc/x", " ".join(sent(40, f"k{i}") for i in range(6)), target_words=50, min_words=0)
    assert [c.id for c in chunks] == [f"doc/x#c{i}" for i in range(len(chunks))]
    assert chunk_text("doc/x", " ".join(sent(40, f"k{i}") for i in range(6)), target_words=50, min_words=0) == chunks


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        chunk_text("d", "x.", target_words=0)
    with pytest.raises(ValueError):
        chunk_text("d", "x.", min_words=-1)


def test_gold_chunk_finds_the_chunk_containing_a_sentence_whatever_the_whitespace():
    text = "Rule one applies to everyone. " + sent(80, "pad") + " The fee is waived for premium holders."
    chunks = chunk_text("d", text, target_words=60, min_words=0)
    hit = gold_chunk(chunks, "The  fee is waived\nfor premium holders")
    assert isinstance(hit, Chunk) and "waived for premium holders" in hit.text
    assert gold_chunk(chunks, "A sentence that is nowhere in the document.") is None
    assert gold_chunk(chunks, "   ") is None
