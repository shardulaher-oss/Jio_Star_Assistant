import pytest

from contract_intelligence.config import ChunkingConfig
from contract_intelligence.indexing.chunking import chunk_document
from contract_intelligence.indexing.segmentation import segment, split_long_unit
from contract_intelligence.schemas import Document


def units(text: str) -> list[str]:
    return [text[a:b] for a, b in segment(text)]


def test_splits_sentences_and_paragraphs() -> None:
    text = "The term is two years. Either party may renew.\n\nSection 2 applies."
    assert units(text) == ["The term is two years.", "Either party may renew.", "Section 2 applies."]


def test_wide_space_runs_are_paragraph_breaks() -> None:
    assert units("1. Term   The term of this Agreement") == ["1. Term", "The term of this Agreement"]


@pytest.mark.parametrize(
    "text",
    [
        "Glu Mobile Inc. shall pay the fee.",
        "Approved by R. C. Boyd Enterprises.",
        "As set out in Section 4. The parties agree.",
        "Delivered in the U.S. Market only.",
        "See clause 9.2. Notice shall be given.",
    ],
)
def test_abbreviations_initials_and_clause_numbers_do_not_split(text: str) -> None:
    assert units(text) == [text]


def test_lowercase_after_period_does_not_split() -> None:
    text = "payable within 30 days. thereafter interest accrues."
    assert units(text) == [text]


def test_units_cover_all_non_whitespace_and_keep_offsets() -> None:
    text = '  "Term." The term ends 2025.\n\n\n  (a) Renewal? Yes!   Done.  '
    spans = segment(text)
    covered = {i for a, b in spans for i in range(a, b)}
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())
    assert all(text[a:b] == text[a:b].strip() for a, b in spans)


def test_split_long_unit_breaks_at_whitespace() -> None:
    text = "word " * 50
    pieces = split_long_unit(text, 0, len(text.rstrip()), 40)
    assert all(b - a <= 40 for a, b in pieces)
    assert all(text[a:b].split() == text[a:b].strip().split() for a, b in pieces)
    assert "".join(text[a:b] for a, b in pieces).replace(" ", "") == text.replace(" ", "")


DOC = Document(
    "cuad/x.txt",
    " ".join(f"Sentence number {i} states a separate obligation of the parties." for i in range(40)),
)


@pytest.mark.parametrize("size,overlap", [(200, 0), (200, 80), (500, 150), (90, 0)])
def test_sentence_chunks_start_and_end_on_sentence_boundaries(size: int, overlap: int) -> None:
    boundaries = segment(DOC.text)
    starts, ends = {a for a, _ in boundaries}, {b for _, b in boundaries}
    chunks = chunk_document(DOC, ChunkingConfig("sentence", size, overlap))
    for chunk in chunks:
        assert DOC.text[chunk.start : chunk.end] == chunk.text
        assert chunk.start in starts and chunk.end in ends
        assert chunk.end - chunk.start <= size
    assert chunks[-1].end == len(DOC.text)


def test_sentence_overlap_repeats_whole_trailing_sentences() -> None:
    no_overlap = chunk_document(DOC, ChunkingConfig("sentence", 200, 0))
    overlap = chunk_document(DOC, ChunkingConfig("sentence", 200, 80))
    assert all(b.start == a.end + 1 for a, b in zip(no_overlap, no_overlap[1:]))
    assert all(b.start < a.end for a, b in zip(overlap, overlap[1:]))
    assert len(overlap) > len(no_overlap)


def test_unit_longer_than_chunk_size_is_split() -> None:
    doc = Document("cuad/y.txt", "x " * 400)
    chunks = chunk_document(doc, ChunkingConfig("sentence", 100, 0))
    assert chunks and all(c.end - c.start <= 100 for c in chunks)
