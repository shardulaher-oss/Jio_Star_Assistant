"""The single retrieval backend shared by the CLI, the web app and evaluation."""

from __future__ import annotations

from contract_intelligence.config import PipelineConfig, RetrievalConfig
from contract_intelligence.indexing.dense_index import (
    DenseIndex,
    Encoder,
    SentenceTransformerEncoder,
    load_or_build,
)
from contract_intelligence.schemas import Passage, SearchResult


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
    def __init__(self, config: PipelineConfig, encoder: Encoder, dense: DenseIndex) -> None:
        self.config = config
        self.encoder = encoder
        self.dense = dense

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Retriever:
        encoder = SentenceTransformerEncoder(config.embedding)
        return cls(config, encoder, load_or_build(config, encoder))

    def candidates(self, queries: list[str], k: int | None = None) -> list[list[Passage]]:
        """Ranked passages per query, before any abstention decision."""
        k = k or self.config.retrieval.top_k
        scores, positions = self.dense.search(self.encoder.encode_queries(queries), k)
        results = []
        for score_row, position_row in zip(scores, positions):
            passages = []
            for score, position in zip(score_row, position_row):
                if position < 0:  # FAISS pads with -1 when fewer than k vectors exist
                    continue
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
