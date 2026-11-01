from dataclasses import replace

import faiss
import pytest
from fakes import HashingEncoder

from contract_intelligence.config import PipelineConfig, RetrievalConfig
from contract_intelligence.indexing.dense_index import DenseIndex
from contract_intelligence.retrieval.retriever import Retriever, decide
from contract_intelligence.schemas import Chunk, Passage

TEXTS = [
    "This Agreement shall be governed by the laws of the State of New York.",
    "Either party may terminate this Agreement for convenience on thirty days notice.",
    "Licensee may not sublicense any rights granted hereunder without prior written consent.",
]


def make_retriever(threshold: float | None = None, top_k: int = 2) -> Retriever:
    encoder = HashingEncoder()
    chunks = [
        Chunk(f"doc:{i}", "cuad/a.txt", i, i * 100, i * 100 + len(t), t) for i, t in enumerate(TEXTS)
    ]
    index = faiss.IndexFlatIP(encoder.dimension)
    index.add(encoder.encode_passages(TEXTS))
    config = replace(
        PipelineConfig(name="test"),
        retrieval=RetrievalConfig(top_k=top_k, abstain_threshold=threshold),
    )
    return Retriever(config, encoder, DenseIndex(index, chunks, {}))


def passage(score: float) -> Passage:
    return Passage(1, "id", "cuad/a.txt", 0, 10, score, "text")


def test_results_are_ranked_with_offsets_and_scores() -> None:
    result = make_retriever().search("governed by the laws of which state")
    assert not result.abstained
    assert [p.rank for p in result.passages] == [1, 2]
    top = result.passages[0]
    assert top.text == TEXTS[0] and (top.start, top.end) == (0, len(TEXTS[0]))
    assert result.passages[0].score >= result.passages[1].score


def test_abstains_with_empty_result_below_threshold() -> None:
    result = make_retriever(threshold=0.99).search("weather forecast for tomorrow")
    assert result.abstained and result.passages == []
    assert result.top_score is not None and result.top_score < 0.99
    assert "below the threshold" in result.message


def test_no_threshold_never_abstains_on_scores() -> None:
    assert decide([passage(-0.5)], RetrievalConfig(abstain_threshold=None)) == (False, "Returned 1 passages.")


def test_threshold_boundary_is_inclusive() -> None:
    assert decide([passage(0.6)], RetrievalConfig(abstain_threshold=0.6))[0] is False
    assert decide([passage(0.5999)], RetrievalConfig(abstain_threshold=0.6))[0] is True


def test_empty_query_abstains_without_searching() -> None:
    results = make_retriever().search_many(["   ", "terminate for convenience"])
    assert results[0].abstained and results[0].message == "The query is empty."
    assert results[1].passages[0].text == TEXTS[1]


def test_top_k_larger_than_index_is_capped() -> None:
    assert len(make_retriever(top_k=50).search("agreement").passages) == len(TEXTS)


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=0)
