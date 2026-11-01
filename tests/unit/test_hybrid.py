import faiss
import numpy as np
from fakes import HashingEncoder

from contract_intelligence.config import HybridConfig
from contract_intelligence.indexing.dense_index import DenseIndex
from contract_intelligence.retrieval.hybrid_ranker import HybridRanker
from contract_intelligence.schemas import Chunk

TEXTS = [
    "This Agreement shall be governed by the laws of the State of New York.",
    "Either party may terminate this Agreement for convenience on thirty days notice.",
    "Licensee shall not sublicense the Licensed Content without written consent.",
    "Payments are due within thirty days of invoice.",
]


class KeywordReranker:
    """Fake cross-encoder: scores a passage by how many times it contains a keyword."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls = 0

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([p.lower().count(self.keyword) for p in passages], dtype=np.float32)


def make(config: HybridConfig, reranker=None) -> HybridRanker:
    encoder = HashingEncoder()
    chunks = [Chunk(f"c{i}", "cuad/a.txt", i, i * 100, i * 100 + len(t), t) for i, t in enumerate(TEXTS)]
    index = faiss.IndexFlatIP(encoder.dimension)
    index.add(encoder.encode_passages(TEXTS))
    return HybridRanker(encoder, DenseIndex(index, chunks, {}), config, reranker)


def test_fusion_returns_union_of_both_pools_with_rrf_scores() -> None:
    ranked = make(HybridConfig(dense_pool=2, bm25_pool=2)).rank(["sublicense licensed content"], k=10)[0]
    assert ranked[0][0] == 2
    assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:]))
    assert 0 < ranked[0][1] <= 2 / 61


def test_reranker_reorders_only_the_pool_and_keeps_the_tail_below() -> None:
    reranker = KeywordReranker("thirty")
    ranked = make(HybridConfig(dense_pool=4, bm25_pool=4, rerank_pool=3), reranker).rank(
        ["governed by the laws of new york"], k=10
    )[0]
    pool_scores = [s for _, s in ranked[:3]]
    assert pool_scores == sorted(pool_scores, reverse=True)
    assert all(s < min(pool_scores) for _, s in ranked[3:])
    assert reranker.calls == 1


def test_question_mode_drops_the_contract_prefix() -> None:
    reranker = KeywordReranker("x")
    seen = []
    reranker.score = lambda q, p: seen.append(q) or np.zeros(len(p), dtype=np.float32)
    make(HybridConfig(rerank_query="question"), reranker).rank(
        ["Consider the Starlight License; who may sublicense?"], k=2
    )
    assert seen == ["who may sublicense?"]


def test_sentence_max_lifts_a_chunk_whose_one_sentence_matches() -> None:
    from contract_intelligence.retrieval.reranking import score_pool

    class SentenceKeyword:
        def score(self, query, passages):
            # A short exact match scores high; long text is diluted, like a real cross-encoder.
            return np.array([10.0 / len(p) if "not assignable" in p else 0.0 for p in passages])

    boiler = "The terms shall bind and benefit the parties and their successors and assigns hereunder. "
    texts = [boiler * 5 + "The rights under this Agreement are not assignable by any Party.", boiler * 6]
    chunk = score_pool(SentenceKeyword(), "q", texts, "chunk")
    best = score_pool(SentenceKeyword(), "q", texts, "sentence_max")
    assert best[0] > chunk[0] and best[1] == chunk[1] == 0.0


def test_chunk_granularity_is_unchanged_behaviour() -> None:
    reranker = KeywordReranker("thirty")
    ranked = make(HybridConfig(dense_pool=4, bm25_pool=4, rerank_pool=3), reranker).rank(["governed by"], k=4)[0]
    assert reranker.calls == 1 and len(ranked) == 4
