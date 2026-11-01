"""Summarise where the current configuration fails, from a scored TRAIN run.

Gold spans enter only through the per-query scores computed after retrieval; they
are never passed to the retriever.
"""

from __future__ import annotations

from collections import Counter

PRIMARY_K = 5


def at_k(values: dict, k: int) -> float:
    """Metric at k from a per-query dict, whether keys are ints (in memory) or strings (JSON)."""
    return values[k] if k in values else values[str(k)]


def diagnose(per_query: list[dict]) -> dict[str, int]:
    """Failure counts used by the proposer (answerable failures at k = PRIMARY_K)."""
    counts: Counter = Counter()
    for q in per_query:
        if not q["answerable"]:
            if not q["abstained"]:
                counts["false_answer"] += 1
                counts[f"false_answer:{q['category']}"] += 1
            continue
        if q["abstained"]:
            counts["false_abstention"] += 1
        rank = q["first_relevant_rank"]
        recall = at_k(q["recall"], PRIMARY_K)
        top_files = {r[0] for r in q["retrieved_top"][:PRIMARY_K]}
        if not top_files & {g[0] for g in q["gold"]}:
            counts["wrong_document"] += 1
        elif rank is None or rank > PRIMARY_K:
            counts["ranked_too_low" if rank is not None else "not_retrieved"] += 1
        elif recall < 0.99:
            counts["partial_coverage"] += 1
        else:
            counts["full_coverage"] += 1
    return dict(counts)
