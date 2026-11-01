"""Character-level precision and recall, as implemented by the LegalBench-RAG authors.

`precision` and `recall` mirror QAResult.precision / QAResult.recall in
github.com/zeroentropy-ai/legalbenchrag (legalbenchrag/run_benchmark.py):

    precision = sum over retrieved spans of |overlap with gold| / sum of retrieved lengths
    recall    = sum over gold spans of |overlap with retrieved| / sum of gold lengths

Overlaps are summed per retrieved span without merging, so overlapping retrieved
chunks count shared characters twice (recall can exceed 1). An empty retrieval
scores 0 for both. The `*_merged` variants merge retrieved spans per file first;
they equal the official numbers when chunks do not overlap and guard against
overlap inflating the official ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class Span(NamedTuple):
    file_path: str
    start: int
    end: int


def _overlap(a: Span, b: Span) -> int:
    if a.file_path != b.file_path:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def precision(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    total = sum(r.end - r.start for r in retrieved)
    if total == 0:
        return 0.0
    return sum(_overlap(r, g) for r in retrieved for g in gold) / total


def recall(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    total = sum(g.end - g.start for g in gold)
    if total == 0:
        return 0.0
    return sum(_overlap(r, g) for g in gold for r in retrieved) / total


def merge_spans(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans):
        if merged and merged[-1].file_path == span.file_path and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(last.file_path, last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return merged


def precision_merged(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    return precision(merge_spans(retrieved), gold)


def recall_merged(retrieved: Sequence[Span], gold: Sequence[Span]) -> float:
    return recall(merge_spans(retrieved), gold)


def first_relevant_rank(retrieved: Sequence[Span], gold: Sequence[Span]) -> int | None:
    """1-based rank of the first retrieved span that overlaps any gold span."""
    for rank, span in enumerate(retrieved, 1):
        if any(_overlap(span, g) for g in gold):
            return rank
    return None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
