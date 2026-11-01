"""Cross-encoder reranking.

A bi-encoder embeds query and passage separately; a cross-encoder reads the pair
together, so it can model which words of the clause answer which words of the
question. It is far more accurate per pair and far slower, so it only rescores a
short candidate pool. Scores are raw relevance logits (higher is more relevant),
not probabilities.

Scores are cached on disk per (model, query, passage), so re-evaluating a
configuration reproduces the same numbers without rerunning the model, and the
improvement loop only pays for pairs it has not scored before.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


MIN_SENTENCE_CHARS = 40  # shorter units (headings, numbers) are scored only as part of the chunk


def sentence_units(text: str) -> list[str]:
    from contract_intelligence.indexing.segmentation import segment

    return [text[a:b] for a, b in segment(text) if b - a >= MIN_SENTENCE_CHARS]


def score_pool(reranker: Reranker, query: str, texts: list[str], granularity: str = "chunk") -> np.ndarray:
    """Relevance score per chunk.

    chunk:        cross-encoder score of the whole chunk.
    sentence_max: max(chunk score, best score of any sentence in the chunk). A
                  one-sentence answer inside a long chunk of boilerplate is then not
                  diluted by the rest of the chunk. All pairs go in one batch.
    """
    if granularity == "chunk" or not texts:
        return reranker.score(query, texts)
    units = [sentence_units(t) for t in texts]
    flat = list(texts) + [u for group in units for u in group]
    scores = reranker.score(query, flat)
    out, offset = scores[: len(texts)].copy(), len(texts)
    for i, group in enumerate(units):
        if group:
            out[i] = max(out[i], float(scores[offset : offset + len(group)].max()))
            offset += len(group)
    return out


def rerank_fused(
    reranker: Reranker, query: str, fused: list[tuple[int, float]], texts_of, pool_size: int, granularity: str
) -> list[tuple[int, float]]:
    """Rescore the top `pool_size` fused candidates; the rest follow below, in fused order."""
    pool = fused[:pool_size]
    scores = score_pool(reranker, query, [texts_of(p) for p, _ in pool], granularity)
    reranked = sorted(zip((p for p, _ in pool), map(float, scores)), key=lambda x: (-x[1], x[0]))
    floor = min((s for _, s in reranked), default=0.0)
    tail = [(p, floor - 1.0 - i * 1e-6) for i, (p, _) in enumerate(fused[pool_size:])]
    return reranked + tail


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        max_length: int = 512,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._model = None
        self.cache: dict[str, float] = {}
        self.cache_path = None
        if cache_dir is not None:
            safe = model_name.replace("/", "__")
            self.cache_path = Path(cache_dir) / f"rerank_{safe}.jsonl"
            if self.cache_path.exists():
                for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                    key, value = json.loads(line)
                    self.cache[key] = value

    @property
    def model(self):
        if self._model is None:  # load lazily: a fully cached run never needs the model
            import torch
            from sentence_transformers import CrossEncoder

            # float32: some checkpoints default to float16, which is ~2x slower on CPU.
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length,
                                       model_kwargs={"dtype": torch.float32})
        return self._model

    def _key(self, query: str, passage: str) -> str:
        return hashlib.sha1(f"{self.model_name}\x00{query}\x00{passage}".encode("utf-8")).hexdigest()

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        texts = [" ".join(p.split()) for p in passages]
        keys = [self._key(query, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            scores = self.model.predict(
                [(query, texts[i]) for i in missing], batch_size=self.batch_size, show_progress_bar=False
            )
            new = {keys[i]: float(s) for i, s in zip(missing, scores)}
            self.cache.update(new)
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    for key, value in new.items():
                        handle.write(json.dumps([key, value]) + "\n")
        return np.array([self.cache[k] for k in keys], dtype=np.float32)
