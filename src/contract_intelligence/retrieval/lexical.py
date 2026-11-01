"""Keyword retrieval: BM25 over the dense index's chunks (pipeline: bm25)."""

from __future__ import annotations

from contract_intelligence.config import HybridConfig
from contract_intelligence.indexing.bm25_index import BM25Index
from contract_intelligence.indexing.dense_index import DenseIndex
from contract_intelligence.retrieval.query import parse_query


def query_text(query: str, mode: str) -> str:
    """'full' keeps the query as typed; 'question' drops the "Consider the ...;" prefix."""
    return parse_query(query).question if mode == "question" else query


class BM25Ranker:
    def __init__(self, dense: DenseIndex, config: HybridConfig) -> None:
        self.config = config
        self.index = BM25Index([c.text for c in dense.chunks], k1=config.bm25_k1, b=config.bm25_b)

    def rank(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        return [self.index.search(query_text(q, self.config.bm25_query), k) for q in queries]
