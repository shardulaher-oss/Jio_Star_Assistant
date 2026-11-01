from pathlib import Path

import pytest

from contract_intelligence.config import ChunkingConfig, load_config
from contract_intelligence.data.corpus import load_documents
from contract_intelligence.indexing.chunking import chunk_corpus, chunk_document
from contract_intelligence.schemas import Document

ROOT = Path(__file__).resolve().parents[2]
TEXT = "Section 1. Term.\n\n   The term is five years.   \n\nSection 2. Law.  " * 20
DOC = Document(file_path="cuad/example.txt", text=TEXT)


@pytest.mark.parametrize("size,overlap", [(50, 0), (50, 10), (37, 36), (1000, 0)])
def test_chunk_text_matches_original_offsets(size: int, overlap: int) -> None:
    for chunk in chunk_document(DOC, ChunkingConfig(chunk_size=size, chunk_overlap=overlap)):
        assert TEXT[chunk.start : chunk.end] == chunk.text
        assert chunk.text == chunk.text.strip() and chunk.text
        assert chunk.end - chunk.start <= size


def test_chunks_cover_every_non_whitespace_character() -> None:
    chunks = chunk_document(DOC, ChunkingConfig(chunk_size=50, chunk_overlap=0))
    covered = {i for c in chunks for i in range(c.start, c.end)}
    assert all(i in covered for i, ch in enumerate(TEXT) if not ch.isspace())


def test_overlap_makes_consecutive_windows_share_text() -> None:
    chunks = chunk_document(DOC, ChunkingConfig(chunk_size=100, chunk_overlap=40))
    assert all(nxt.start < cur.end for cur, nxt in zip(chunks, chunks[1:]))


def test_chunk_ids_are_unique_and_stable() -> None:
    config = ChunkingConfig(chunk_size=60, chunk_overlap=20)
    first = [c.chunk_id for c in chunk_document(DOC, config)]
    assert len(set(first)) == len(first)
    assert first == [c.chunk_id for c in chunk_document(DOC, config)]


def test_whitespace_only_document_yields_no_chunks() -> None:
    assert chunk_document(Document("cuad/blank.txt", " \n\t "), ChunkingConfig()) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunking_config_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=size, chunk_overlap=overlap)


@pytest.mark.skipif(
    not (ROOT / "data/raw/corpus/cuad").exists(), reason="corpus not downloaded"
)
def test_real_corpus_chunks_preserve_offsets() -> None:
    config = load_config(ROOT / "configs/baseline.yaml")
    documents = load_documents(
        config.resolve(config.data.document_manifest), config.resolve(config.data.corpus_root)
    )
    texts = {d.file_path: d.text for d in documents}
    chunks = chunk_corpus(documents, config.chunking)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(texts[c.file_path][c.start : c.end] == c.text for c in chunks)
