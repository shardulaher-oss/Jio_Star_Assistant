"""Load and validate pipeline configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINES = ("dense", "graph", "bm25", "hybrid", "graph_hybrid")
QUERY_MODES = ("full", "question")
ANSWER_PROVIDERS = ("gemini",)


class ConfigError(ValueError):
    """A configuration file is missing, malformed or has invalid values."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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

    def __post_init__(self) -> None:
        _check(0 < self.route_keep_ratio <= 1, "route_keep_ratio must be in (0, 1]")
        _check(self.route_min_score >= 0, "route_min_score must be >= 0")
        _check(self.expand_top >= 0, "expand_top must be >= 0")
        _check(0 <= self.expansion_decay <= 1, "expansion_decay must be in [0, 1]")


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
    rerank_granularity: str = "chunk"  # chunk | sentence_max (best of chunk and its sentences)
    # graph_hybrid only: the graph ranking as an extra RRF input (0 disables a list)
    graph_pool: int = 50
    graph_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("bm25_query", "dense_query", "rerank_query"):
            _check(getattr(self, name) in QUERY_MODES, f"{name} must be one of {QUERY_MODES}")
        _check(self.rerank_granularity in ("chunk", "sentence_max"),
               "rerank_granularity must be 'chunk' or 'sentence_max'")
        for name in ("dense_pool", "bm25_pool", "graph_pool"):  # 0 = that list contributes nothing
            _check(getattr(self, name) >= 0, f"{name} must be >= 0")
        for name in ("rerank_pool", "rrf_k", "rerank_batch_size"):
            _check(getattr(self, name) > 0, f"{name} must be positive")
        for name in ("dense_weight", "bm25_weight", "graph_weight"):
            _check(getattr(self, name) >= 0, f"{name} must be >= 0")
        _check(self.bm25_k1 >= 0 and 0 <= self.bm25_b <= 1, "bm25_k1 must be >= 0 and bm25_b in [0, 1]")


@dataclass(frozen=True)
class RetrievalConfig:
    pipeline: str = "dense"  # dense | graph | bm25 | hybrid | graph_hybrid
    top_k: int = 10
    # Abstain when the best cosine score is below this. A similarity score, not a
    # probability: it is only meaningful relative to scores from the same model.
    abstain_threshold: float | None = None

    def __post_init__(self) -> None:
        _check(self.pipeline in PIPELINES, f"pipeline must be one of {PIPELINES}")
        _check(self.top_k > 0, "top_k must be positive")


@dataclass(frozen=True)
class AnswerConfig:
    """Optional LLM answer on top of retrieval. Off unless requested (CLI --answer, app toggle)."""

    provider: str = "gemini"
    # gemini-2.5-flash is closed to new API keys; 3.8-flash timed out in testing (Nov 2026).
    model: str = "gemini-3.1-flash-lite"
    max_passages: int = 5  # top retrieved passages shown to the model
    temperature: float = 0.0
    thinking_budget: int | None = None  # Gemini 2.5 only; None leaves the model default
    timeout_seconds: int = 60
    cache_path: str = "experiments/llm_cache/answers.jsonl"  # committed, so results reproduce offline
    # Privacy switch: the provider is an external API, so passages (contract text) leave the
    # machine. True is acceptable for the public CUAD corpus; set False for confidential
    # agreements and answer generation will refuse to send anything.
    allow_external: bool = True

    def __post_init__(self) -> None:
        _check(self.provider in ANSWER_PROVIDERS, f"provider must be one of {ANSWER_PROVIDERS}")
        _check(1 <= self.max_passages <= 20, "max_passages must be in [1, 20]")
        _check(0 <= self.temperature <= 2, "temperature must be in [0, 2]")
        _check(self.timeout_seconds > 0, "timeout_seconds must be positive")


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
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    def resolve(self, relative: str) -> Path:
        return PROJECT_ROOT / relative


SECTIONS = {
    "data": DataConfig, "chunking": ChunkingConfig, "embedding": EmbeddingConfig,
    "index": IndexConfig, "retrieval": RetrievalConfig, "graph": GraphConfig,
    "hybrid": HybridConfig, "answer": AnswerConfig,
}


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config, rejecting unknown keys and invalid values with a clear message."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    if not isinstance(raw.get("name"), str) or not raw["name"]:
        raise ConfigError(f"{path}: 'name' (a non-empty string) is required")
    unknown = set(raw) - set(SECTIONS) - {"name"}
    if unknown:
        raise ConfigError(f"{path}: unknown section(s) {sorted(unknown)}; valid: {sorted(SECTIONS)}")

    sections = {}
    for section, cls in SECTIONS.items():
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"{path}: section '{section}' must be a mapping")
        valid = set(cls.__dataclass_fields__)
        unknown_keys = set(values) - valid
        if unknown_keys:
            raise ConfigError(f"{path}: unknown key(s) in '{section}': {sorted(unknown_keys)}; "
                              f"valid: {sorted(valid)}")
        try:
            sections[section] = cls(**values)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{path}: invalid '{section}' settings: {error}") from error
    return PipelineConfig(name=raw["name"], **sections)
