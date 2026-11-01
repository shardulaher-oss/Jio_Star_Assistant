"""Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

    rrf(d) = sum over rankings r containing d of  weight_r / (k + rank_r(d))

It uses ranks only, so BM25 scores (unbounded) and cosine similarities ([-1, 1])
can be combined without calibrating one against the other. A larger k flattens the
contribution of the top ranks; k = 60 is the value from the original paper.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids (best first) into one list of (id, rrf score), best first."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, 1):
            fused[item] += weight / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
