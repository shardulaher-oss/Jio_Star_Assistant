import numpy as np


class HashingEncoder:
    """Deterministic bag-of-words encoder, so tests need no model download."""

    dimension = 256

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                vectors[row, hash(token) % self.dimension] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    encode_passages = _encode
    encode_queries = _encode
