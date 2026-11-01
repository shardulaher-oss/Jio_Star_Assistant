"""Hybrid RAG: BM25 + dense candidates, Reciprocal Rank Fusion, optional reranking.

1. Dense (FAISS) pool of `dense_pool` chunks and BM25 pool of `bm25_pool` chunks.
2. Fuse the two rankings with RRF (`rrf_k`).
3. If a reranker is configured, rescore the top `rerank_pool` fused chunks with the
   cross-encoder and order them by its score. Fused chunks beyond the pool follow
   in RRF order, below every reranked chunk.
The returned score is the cross-encoder logit when reranking, else the RRF score;
the abstention threshold must be calibrated for whichever is used.
"""

from __future__ import annotations

from contract_intelligence.config import HybridConfig
from contract_intelligence.indexing.dense_index import DenseIndex, Encoder
from contract_intelligence.retrieval.fusion import reciprocal_rank_fusion
from contract_intelligence.retrieval.lexical import BM25Ranker, query_text
from contract_intelligence.retrieval.reranking import Reranker, rerank_fused


class HybridRanker:
    def __init__(
        self, encoder: Encoder, dense: DenseIndex, config: HybridConfig, reranker: Reranker | None = None
    ) -> None:
        self.encoder = encoder
        self.dense = dense
        self.config = config
        self.bm25 = BM25Ranker(dense, config)
        self.reranker = reranker

    def fused(self, queries: list[str]) -> list[list[tuple[int, float]]]:
        cfg = self.config
        dense_texts = [query_text(q, cfg.dense_query) for q in queries]
        _, positions = self.dense.search(self.encoder.encode_queries(dense_texts), cfg.dense_pool)
        lexical = self.bm25.rank(queries, cfg.bm25_pool)
        return [
            reciprocal_rank_fusion(
                [[int(p) for p in dense_row if p >= 0], [p for p, _ in lexical_row]],
                k=cfg.rrf_k,
                weights=[cfg.dense_weight, cfg.bm25_weight],
            )
            for dense_row, lexical_row in zip(positions, lexical)
        ]

    def rerank(self, query: str, fused: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return rerank_fused(
            self.reranker, query_text(query, self.config.rerank_query), fused,
            lambda p: self.dense.chunks[p].text, self.config.rerank_pool, self.config.rerank_granularity,
        )

    def rank(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        results = []
        for query, fused in zip(queries, self.fused(queries)):
            ranked = self.rerank(query, fused) if self.reranker is not None else fused
            results.append(ranked[:k])
        return results
