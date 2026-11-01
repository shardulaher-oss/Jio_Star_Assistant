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
    strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        if self.strategy != "fixed":
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    data: DataConfig = field(default_factory=DataConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def resolve(self, relative: str) -> Path:
        return PROJECT_ROOT / relative


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig(
        name=raw["name"],
        data=DataConfig(**raw.get("data", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
    )
