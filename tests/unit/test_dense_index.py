from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fakes import HashingEncoder

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


class CountingModel:
    def __init__(self) -> None:
        self.encoded = 0

    def encode(self, inputs, **kwargs):
        self.encoded += len(inputs)
        return np.array([[float(len(t)), 1.0] for t in inputs])


def make_encoder(cache_size: int = 4096):
    from collections import OrderedDict

    from contract_intelligence.indexing.dense_index import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    encoder.config, encoder.model, encoder.dimension = EmbeddingConfig(), CountingModel(), 2
    encoder._query_cache, encoder.query_cache_size, encoder.query_cache_hits = OrderedDict(), cache_size, 0
    return encoder


def test_query_cache_reuses_vectors_and_keeps_order() -> None:
    encoder = make_encoder()
    first = encoder.encode_queries(["a", "bb", "a"])
    second = encoder.encode_queries(["bb", "ccc"])
    assert encoder.model.encoded == 3  # "a", "bb", "ccc" each encoded once
    assert first[0].tolist() == first[2].tolist() and second[0].tolist() == first[1].tolist()
    assert encoder.query_cache_hits == 2


def test_query_cache_is_bounded() -> None:
    encoder = make_encoder(cache_size=2)
    encoder.encode_queries(["a", "b", "c"])
    assert list(encoder._query_cache) == ["b", "c"]
    assert encoder.encode_queries([]).shape == (0, 2)
