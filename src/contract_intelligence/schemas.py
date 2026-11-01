"""Typed records shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    file_path: str  # relative to the corpus root, e.g. "cuad/<name>.txt"
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_path: str
    chunk_index: int
    start: int  # inclusive character offset into the original document
    end: int  # exclusive character offset into the original document
    text: str  # always equal to document.text[start:end]


@dataclass(frozen=True)
class Passage:
    rank: int
    chunk_id: str
    file_path: str
    start: int
    end: int
    score: float  # cosine similarity in [-1, 1]; higher is more similar, not a probability
    text: str


@dataclass(frozen=True)
class SearchResult:
    query: str
    passages: list[Passage]  # empty when the system abstains
    abstained: bool
    top_score: float | None
    threshold: float | None
    message: str
