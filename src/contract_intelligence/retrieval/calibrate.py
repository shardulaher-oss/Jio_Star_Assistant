"""Calibrate the abstention threshold on TRAIN queries only.

The top-1 cosine score is a ranking signal, not a probability, so the threshold is
chosen empirically: compare the top-1 score of answerable train queries with that of
the train portion of the unanswerable set, and pick the cut that maximises balanced
accuracy (the mean of "answered when it should" and "abstained when it should").
Balanced accuracy weights both error types equally even though the two sets differ
in size (140 vs 30 queries).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from contract_intelligence.config import load_config
from contract_intelligence.retrieval.retriever import Retriever


def load_train_queries(config) -> tuple[list[str], list[str]]:
    answerable = json.loads(
        config.resolve("data/processed/queries/train.json").read_text(encoding="utf-8")
    )
    negatives = json.loads(
        config.resolve("data/queries/unanswerable.json").read_text(encoding="utf-8")
    )
    return (
        [r["query"] for r in answerable],
        [r["query"] for r in negatives if r["split"] == "train"],
    )


def sweep(pos: np.ndarray, neg: np.ndarray) -> list[dict]:
    """Evaluate each midpoint between sorted distinct scores as a threshold."""
    values = np.unique(np.concatenate([pos, neg]))
    cuts = np.concatenate([[values[0] - 1e-6], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-6]])
    rows = []
    for cut in cuts:
        answered = float(np.mean(pos >= cut))  # 1 - wrong-abstention rate
        abstained = float(np.mean(neg < cut))  # correct-abstention rate
        rows.append({
            "threshold": round(float(cut), 4),
            "answerable_answered": round(answered, 3),
            "unanswerable_abstained": round(abstained, 3),
            "balanced_accuracy": round((answered + abstained) / 2, 3),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    retriever = Retriever.from_config(config)
    pos_queries, neg_queries = load_train_queries(config)
    pos = np.array([p[0].score for p in retriever.candidates(pos_queries, k=1)])
    neg = np.array([p[0].score for p in retriever.candidates(neg_queries, k=1)])

    def describe(x: np.ndarray) -> dict:
        q = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
        return {"n": len(x), **{f"p{p}": round(float(v), 3) for p, v in zip((0, 10, 25, 50, 75, 90, 100), q)}}

    rows = sweep(pos, neg)
    best = max(rows, key=lambda r: (r["balanced_accuracy"], r["answerable_answered"]))
    print(json.dumps({
        "config": config.name,
        "top1_score_answerable_train": describe(pos),
        "top1_score_unanswerable_train": describe(neg),
        "best_by_balanced_accuracy": best,
        # Highest threshold that still answers at least this share of answerable queries.
        "trade_off": [
            max((r for r in rows if r["answerable_answered"] >= target), key=lambda r: r["threshold"])
            for target in (1.0, 0.95, 0.9, 0.8)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
