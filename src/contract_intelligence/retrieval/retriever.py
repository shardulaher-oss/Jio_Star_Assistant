"""The single retrieval backend shared by the CLI, the web app and evaluation."""

from __future__ import annotations

from typing import Protocol

from contract_intelligence.config import PipelineConfig, RetrievalConfig
from contract_intelligence.indexing.dense_index import (
    DenseIndex,
    Encoder,
    SentenceTransformerEncoder,
    load_or_build,
)
from contract_intelligence.schemas import Passage, SearchResult


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
    raise ValueError(f"Unknown retrieval pipeline: {pipeline}")


def decide(passages: list[Passage], config: RetrievalConfig) -> tuple[bool, str]:
    """Return (abstain, message) for a ranked candidate list."""
    if not passages:
        return True, "No passages were retrieved."
    if config.abstain_threshold is not None and passages[0].score < config.abstain_threshold:
        return True, (
            f"No passage is relevant enough: best score {passages[0].score:.3f} is below "
            f"the threshold {config.abstain_threshold:.3f}."
        )
    return False, f"Returned {len(passages)} passages."


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
        results = []
        cleaned = [q.strip() for q in queries]
        non_empty = [q for q in cleaned if q]
        ranked = iter(self.candidates(non_empty, k) if non_empty else [])
        for query in cleaned:
            if not query:
                results.append(SearchResult(query, [], True, None, None, "The query is empty."))
                continue
            passages = next(ranked)
            abstain, message = decide(passages, self.config.retrieval)
            results.append(SearchResult(
                query=query,
                passages=[] if abstain else passages,
                abstained=abstain,
                top_score=passages[0].score if passages else None,
                threshold=self.config.retrieval.abstain_threshold,
                message=message,
            ))
        return results

    def search(self, query: str, k: int | None = None) -> SearchResult:
        return self.search_many([query], k)[0]
