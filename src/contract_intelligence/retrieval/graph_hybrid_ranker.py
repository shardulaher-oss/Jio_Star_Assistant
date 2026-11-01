"""Graph-enhanced Hybrid RAG (pipeline: graph_hybrid).

Query -> query processing -> graph contract routing -> three candidate lists:
  1. graph ranking: question-only dense + section-title boost + edge expansion,
     inside the routed contracts (GraphRanker);
  2. BM25 on the question, restricted to the routed contracts;
  3. FAISS on the full query over the whole corpus (recovers from a wrong route);
-> weighted RRF -> optional cross-encoder reranking of the top `rerank_pool` on the
question -> abstention (in Retriever).
A list whose weight is 0 is skipped entirely.
"""

from __future__ import annotations

from contract_intelligence.config import GraphConfig, HybridConfig
from contract_intelligence.indexing.bm25_index import BM25Index
from contract_intelligence.indexing.dense_index import DenseIndex, Encoder
from contract_intelligence.retrieval.fusion import reciprocal_rank_fusion
from contract_intelligence.retrieval.graph_ranker import GraphRanker
from contract_intelligence.retrieval.lexical import query_text
from contract_intelligence.retrieval.query import parse_query
from contract_intelligence.retrieval.reranking import Reranker, rerank_fused


class GraphHybridRanker:
    def __init__(
        self,
        encoder: Encoder,
        dense: DenseIndex,
        graph_ranker: GraphRanker,
        hybrid: HybridConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self.encoder = encoder
        self.dense = dense
        self.graph = graph_ranker
        self.config = hybrid
        self.bm25 = BM25Index([c.text for c in dense.chunks], k1=hybrid.bm25_k1, b=hybrid.bm25_b)
        self.reranker = reranker

    def fused(self, queries: list[str]) -> list[list[tuple[int, float]]]:
        cfg = self.config
        graph_lists = self.graph.rank(queries, cfg.graph_pool) if cfg.graph_weight else [[] for _ in queries]
        if cfg.dense_weight:
            texts = [query_text(q, cfg.dense_query) for q in queries]
            _, dense_rows = self.dense.search(self.encoder.encode_queries(texts), cfg.dense_pool)
        else:
            dense_rows = [[] for _ in queries]
        results = []
        for query, graph_list, dense_row in zip(queries, graph_lists, dense_rows):
            if self.graph.names_unknown_contract(query):
                results.append([])
                continue
            rankings, weights = [], []
            if cfg.graph_weight:
                rankings.append([p for p, _ in graph_list])
                weights.append(cfg.graph_weight)
            if cfg.bm25_weight:
                routed = self.graph.route(query)
                allowed = set(self.graph.candidates(routed)) if routed is not None else None
                lexical = self.bm25.search(query_text(query, cfg.bm25_query), cfg.bm25_pool, allowed)
                rankings.append([p for p, _ in lexical])
                weights.append(cfg.bm25_weight)
            if cfg.dense_weight:
                rankings.append([int(p) for p in dense_row if p >= 0])
                weights.append(cfg.dense_weight)
            results.append(reciprocal_rank_fusion(rankings, k=cfg.rrf_k, weights=weights))
        return results

    def rank(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        results = []
        for query, fused in zip(queries, self.fused(queries)):
            if self.reranker is not None:
                fused = rerank_fused(
                    self.reranker, query_text(query, self.config.rerank_query), fused,
                    lambda p: self.dense.chunks[p].text, self.config.rerank_pool, self.config.rerank_granularity,
                )
            results.append(fused[:k])
        return results
