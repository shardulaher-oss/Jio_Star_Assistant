"""The single retrieval backend shared by the CLI, the web app and evaluation."""

from __future__ import annotations

import logging
import time
from typing import Protocol

from contract_intelligence.config import PipelineConfig
from contract_intelligence.indexing.dense_index import (
    DenseIndex,
    Encoder,
    SentenceTransformerEncoder,
    load_or_build,
)
from contract_intelligence.logging_utils import query_id
from contract_intelligence.retrieval.abstention import clean_query, decide  # noqa: F401  (decide re-exported)
from contract_intelligence.schemas import Passage, SearchResult

log = logging.getLogger(__name__)


class Ranker(Protocol):
    def rank(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        """Per query, up to k (chunk position, score) pairs, best first."""
        ...


class DenseRanker:
    """Traditional RAG: cosine similarity of the full query against every chunk."""

    def __init__(self, encoder: Encoder, dense: DenseIndex) -> None:
        self.encoder = encoder
        self.dense = dense

    def rank(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        scores, positions = self.dense.search(self.encoder.encode_queries(queries), k)
        return [
            [(int(p), float(s)) for s, p in zip(score_row, position_row) if p >= 0]  # FAISS pads with -1
            for score_row, position_row in zip(scores, positions)
        ]


def build_reranker(config: PipelineConfig):
    if not config.hybrid.reranker_model:
        return None
    from contract_intelligence.retrieval.reranking import CrossEncoderReranker

    return CrossEncoderReranker(
        config.hybrid.reranker_model,
        config.hybrid.rerank_batch_size,
        config.embedding.device,
        cache_dir=config.resolve("artifacts/cache"),
    )


def build_ranker(config: PipelineConfig, encoder: Encoder, dense: DenseIndex) -> Ranker:
    pipeline = config.retrieval.pipeline
    if pipeline == "dense":
        return DenseRanker(encoder, dense)
    if pipeline == "graph":
        from contract_intelligence.graph.builder import load_or_build_graph
        from contract_intelligence.retrieval.graph_ranker import GraphRanker

        return GraphRanker(encoder, dense, load_or_build_graph(config), config.graph)
    if pipeline == "bm25":
        from contract_intelligence.retrieval.lexical import BM25Ranker

        return BM25Ranker(dense, config.hybrid)
    if pipeline == "hybrid":
        from contract_intelligence.retrieval.hybrid_ranker import HybridRanker

        return HybridRanker(encoder, dense, config.hybrid, build_reranker(config))
    if pipeline == "graph_hybrid":
        from contract_intelligence.graph.builder import load_or_build_graph
        from contract_intelligence.retrieval.graph_hybrid_ranker import GraphHybridRanker
        from contract_intelligence.retrieval.graph_ranker import GraphRanker

        graph_ranker = GraphRanker(encoder, dense, load_or_build_graph(config), config.graph)
        return GraphHybridRanker(encoder, dense, graph_ranker, config.hybrid, build_reranker(config))
    raise ValueError(f"Unknown retrieval pipeline: {pipeline}")


class Retriever:
    def __init__(
        self, config: PipelineConfig, encoder: Encoder, dense: DenseIndex, ranker: Ranker | None = None
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.dense = dense
        self.ranker = ranker or DenseRanker(encoder, dense)

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Retriever:
        encoder = SentenceTransformerEncoder(config.embedding)
        dense = load_or_build(config, encoder)
        return cls(config, encoder, dense, build_ranker(config, encoder, dense))

    def candidates(self, queries: list[str], k: int | None = None) -> list[list[Passage]]:
        """Ranked passages per query, before any abstention decision."""
        k = k or self.config.retrieval.top_k
        results = []
        for ranked in self.ranker.rank(queries, k):
            passages = []
            for position, score in ranked:
                chunk = self.dense.chunks[position]
                passages.append(Passage(
                    rank=len(passages) + 1,
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    start=chunk.start,
                    end=chunk.end,
                    score=float(score),
                    text=chunk.text,
                ))
            results.append(passages)
        return results

    def search_many(self, queries: list[str], k: int | None = None) -> list[SearchResult]:
        started = time.perf_counter()
        checked = [clean_query(q) for q in queries]
        valid = [text for text, problem in checked if problem is None]
        ranked = iter(self.candidates(valid, k) if valid else [])
        results = []
        for text, problem in checked:
            if problem is not None:  # rejected input: explain instead of searching
                results.append(SearchResult(text, [], True, None, None, problem))
                continue
            passages = next(ranked)
            abstain, message = decide(passages, self.config.retrieval)
            results.append(SearchResult(
                query=text,
                passages=[] if abstain else passages,
                abstained=abstain,
                top_score=passages[0].score if passages else None,
                threshold=self.config.retrieval.abstain_threshold,
                message=message,
            ))
        elapsed = (time.perf_counter() - started) * 1000
        for result in results:
            log.info("search", extra={
                "event": "search", "config": self.config.name, "pipeline": self.config.retrieval.pipeline,
                "query_id": query_id(result.query), "query_chars": len(result.query),
                "abstained": result.abstained, "passages": len(result.passages),
                "top_score": result.top_score, "batch_latency_ms": round(elapsed, 1),
            })
        return results

    def search(self, query: str, k: int | None = None) -> SearchResult:
        return self.search_many([query], k)[0]
