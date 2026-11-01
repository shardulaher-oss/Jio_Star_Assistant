"""Cross-encoder reranking.

A bi-encoder embeds query and passage separately; a cross-encoder reads the pair
together, so it can model which words of the clause answer which words of the
question. It is far more accurate per pair and far slower, so it only rescores a
short candidate pool. Scores are raw relevance logits (higher is more relevant),
not probabilities.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu", max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, " ".join(p.split())) for p in passages]
        return np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), dtype=np.float32
        )
