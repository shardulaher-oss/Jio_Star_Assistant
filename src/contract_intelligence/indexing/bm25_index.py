"""Okapi BM25 over the same chunks as the dense index.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
with idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))  (the non-negative Lucene form).

Lexical matching complements embeddings on contract text: clause vocabulary such as
"governed by", "assign", "sublicense" or "audit" is predictable, and exact party
names and defined terms matter. Building takes well under a second for this corpus,
so the index is rebuilt in memory rather than persisted.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# Deliberately small: negations and modal verbs ("not", "shall", "may") carry meaning
# in contracts and are kept.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "which", "what",
    "there", "these", "those", "under", "any", "does", "do",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((doc_id, tf))
        self.n_docs = len(texts)
        self.avgdl = sum(self.doc_len) / max(1, self.n_docs)
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: str, allowed: set[int] | None = None) -> dict[int, float]:
        """BM25 score for every document sharing a term with the query."""
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                if allowed is not None and doc_id not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                out[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores = self.scores(query, allowed)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
