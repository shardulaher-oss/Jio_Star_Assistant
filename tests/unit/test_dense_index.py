from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contract_intelligence.config import ChunkingConfig, EmbeddingConfig, load_config
from contract_intelligence.indexing.dense_index import (
    DenseIndex,
    build_index,
    index_fingerprint,
    prepare_text,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(ROOT / "configs/baseline.yaml")
HAS_CORPUS = (ROOT / "data/raw/corpus/cuad").exists()


class HashingEncoder:
    """Deterministic bag-of-words encoder, so index tests need no model download."""

    dimension = 256

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                vectors[row, hash(token) % self.dimension] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    encode_passages = _encode
    encode_queries = _encode


def test_prepare_text_collapses_whitespace_only_when_enabled() -> None:
    assert prepare_text("a  \n\n b\tc", True) == "a b c"
    assert prepare_text("a  \n b", False) == "a  \n b"


def test_fingerprint_tracks_index_contents_not_runtime_settings() -> None:
    base = index_fingerprint(CONFIG)
    bigger_chunks = replace(CONFIG, chunking=ChunkingConfig(chunk_size=800))
    other_model = replace(CONFIG, embedding=EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5"))
    bigger_batch = replace(CONFIG, embedding=replace(CONFIG.embedding, batch_size=8))
    assert index_fingerprint(bigger_chunks) != base
    assert index_fingerprint(other_model) != base
    assert index_fingerprint(bigger_batch) == base


@pytest.mark.skipif(not HAS_CORPUS, reason="corpus not downloaded")
def test_each_chunk_is_its_own_nearest_neighbour(tmp_path: Path) -> None:
    dense = build_index(CONFIG, HashingEncoder())
    assert dense.index.ntotal == len(dense.chunks)

    probe = [dense.chunks[i] for i in (0, len(dense.chunks) // 2, len(dense.chunks) - 1)]
    scores, positions = dense.search(HashingEncoder().encode_queries([c.text for c in probe]), k=3)
    for chunk, row, score_row in zip(probe, positions, scores):
        assert dense.chunks[row[0]].text == chunk.text
        assert score_row[0] == pytest.approx(1.0, abs=1e-5)
        assert list(score_row) == sorted(score_row, reverse=True)

    dense.save(tmp_path)
    loaded = DenseIndex.load(tmp_path)
    assert loaded.chunks == dense.chunks
    assert loaded.index.ntotal == dense.index.ntotal
