"""Sentence Transformer embeddings and an exact FAISS index over chunks.

Vectors are L2-normalised, so the inner product computed by IndexFlatIP equals
cosine similarity. The corpus is a few thousand chunks, so exact search is fast
and an approximate index would only add recall loss.

Each index is saved under a fingerprint of everything that changes its contents
(corpus hashes, chunking, model, passage text preparation). Changing retrieval-only
settings such as top-k or the threshold reuses the same index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from contract_intelligence.config import EmbeddingConfig, PipelineConfig, load_config
from contract_intelligence.data.corpus import load_documents
from contract_intelligence.indexing.chunking import chunk_corpus
from contract_intelligence.schemas import Chunk


class Encoder(Protocol):
    dimension: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def prepare_text(text: str, normalize_whitespace: bool) -> str:
    return " ".join(text.split()) if normalize_whitespace else text


class SentenceTransformerEncoder:
    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.dimension = self.model.get_embedding_dimension()
        # Bounded LRU of query vectors: the app, CLI loops and evaluation repeat queries,
        # and re-encoding an identical string always yields the identical vector.
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.query_cache_size = 4096
        self.query_cache_hits = 0

    def _encode(self, texts: list[str], prefix: str, progress: bool) -> np.ndarray:
        inputs = [prefix + prepare_text(t, self.config.normalize_whitespace) for t in texts]
        vectors = self.model.encode(
            inputs,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.config.passage_prefix, progress=len(texts) > 500)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode queries in one batch, reusing cached vectors for repeated strings."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._query_cache))
        if missing:
            for text, vector in zip(missing, self._encode(missing, self.config.query_prefix, progress=False)):
                self._query_cache[text] = vector
        self.query_cache_hits += len(texts) - len(missing)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = np.stack([self._query_cache[t] for t in texts])  # before eviction
        for text in texts:
            self._query_cache.move_to_end(text)
        while len(self._query_cache) > self.query_cache_size:
            self._query_cache.popitem(last=False)
        return vectors


def index_fingerprint(config: PipelineConfig) -> str:
    manifest = config.resolve(config.data.document_manifest).read_bytes()
    key = {
        "corpus": hashlib.sha256(manifest).hexdigest(),
        "chunking": asdict(config.chunking),
        "model_name": config.embedding.model_name,
        "normalize_whitespace": config.embedding.normalize_whitespace,
        "passage_prefix": config.embedding.passage_prefix,
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass
class DenseIndex:
    index: faiss.Index
    chunks: list[Chunk]
    manifest: dict

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, chunk positions), each shaped (n_queries, k), best first."""
        k = min(k, self.index.ntotal)
        return self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> DenseIndex:
        index = faiss.read_index(str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            chunks = [Chunk(**json.loads(line)) for line in handle]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if index.ntotal != len(chunks):
            raise ValueError(f"{directory}: index has {index.ntotal} vectors, {len(chunks)} chunks")
        return cls(index=index, chunks=chunks, manifest=manifest)


def build_index(config: PipelineConfig, encoder: Encoder) -> DenseIndex:
    documents = load_documents(
        config.resolve(config.data.document_manifest), config.resolve(config.data.corpus_root)
    )
    chunks = chunk_corpus(documents, config.chunking)
    vectors = encoder.encode_passages([chunk.text for chunk in chunks])
    index = faiss.IndexFlatIP(encoder.dimension)
    index.add(vectors)
    manifest = {
        "fingerprint": index_fingerprint(config),
        "config_name": config.name,
        "model_name": config.embedding.model_name,
        "dimension": encoder.dimension,
        "chunking": asdict(config.chunking),
        "documents": len(documents),
        "chunks": len(chunks),
    }
    return DenseIndex(index=index, chunks=chunks, manifest=manifest)


def index_dir(config: PipelineConfig) -> Path:
    return config.resolve(config.index.output_root) / index_fingerprint(config)


def load_or_build(
    config: PipelineConfig, encoder: Encoder, rebuild: bool = False
) -> DenseIndex:
    directory = index_dir(config)
    if not rebuild and (directory / "manifest.json").exists():
        return DenseIndex.load(directory)
    dense = build_index(config, encoder)
    dense.save(directory)
    return dense


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and build the FAISS index.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    started = time.perf_counter()
    encoder = SentenceTransformerEncoder(config.embedding)
    dense = load_or_build(config, encoder, rebuild=args.rebuild)
    print(json.dumps({
        **dense.manifest,
        "path": str(index_dir(config)),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
