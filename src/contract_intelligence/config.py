"""Load and validate pipeline configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    corpus_root: str = "data/raw/corpus"
    document_manifest: str = "data/processed/document_manifest.json"


@dataclass(frozen=True)
class ChunkingConfig:
    # "fixed": character windows. "sentence": whole sentences/paragraphs packed up to
    # chunk_size characters; chunk_overlap is the character budget of whole sentences
    # repeated at the start of the next chunk.
    strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "sentence"):
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_whitespace: bool = True  # applied to a copy of the text, never to offsets
    query_prefix: str = ""  # some models (e.g. BGE, E5) expect an instruction prefix
    passage_prefix: str = ""
    batch_size: int = 64
    device: str = "cpu"


@dataclass(frozen=True)
class IndexConfig:
    output_root: str = "artifacts/indexes"


@dataclass(frozen=True)
class GraphConfig:
    output_root: str = "artifacts/graphs"
    # Graph retrieval (pipeline: graph)
    route_contracts: bool = True  # restrict search to contracts the query names
    route_min_score: float = 3.0  # min IDF-weighted name overlap to trust a route
    route_keep_ratio: float = 0.5  # keep contracts scoring >= ratio * best
    question_only: bool = True  # embed the question without the "Consider the ..." prefix
    section_title_weight: float = 0.3  # boost for question vs section-title similarity
    expand_top: int = 5  # seeds whose REFERENCES / DEFINES neighbours are added
    expansion_decay: float = 0.9  # neighbour score = seed score * decay
    # Return nothing when the query names an agreement ("Consider the ...;") that the
    # router cannot match to any indexed contract.
    abstain_when_unrouted: bool = False


@dataclass(frozen=True)
class HybridConfig:
    # Query text per stage: full | question (drop the "Consider the ...;" prefix)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_query: str = "full"
    dense_query: str = "full"
    dense_pool: int = 50  # candidates taken from FAISS
    bm25_pool: int = 50  # candidates taken from BM25
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    reranker_model: str = ""  # e.g. cross-encoder/ms-marco-MiniLM-L6-v2; "" disables
    rerank_pool: int = 40  # fused candidates rescored by the cross-encoder
    rerank_query: str = "full"
    rerank_batch_size: int = 32
    # graph_hybrid only: the graph ranking as an extra RRF input (0 disables a list)
    graph_pool: int = 50
    graph_weight: float = 1.0


@dataclass(frozen=True)
class RetrievalConfig:
    pipeline: str = "dense"  # dense | graph | bm25 | hybrid | graph_hybrid
    top_k: int = 10
    # Abstain when the best cosine score is below this. A similarity score, not a
    # probability: it is only meaningful relative to scores from the same model.
    abstain_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    data: DataConfig = field(default_factory=DataConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    def resolve(self, relative: str) -> Path:
        return PROJECT_ROOT / relative


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig(
        name=raw["name"],
        data=DataConfig(**raw.get("data", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        index=IndexConfig(**raw.get("index", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        graph=GraphConfig(**raw.get("graph", {})),
        hybrid=HybridConfig(**raw.get("hybrid", {})),
    )
