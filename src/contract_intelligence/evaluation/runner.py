"""Evaluate a pipeline configuration on one split.

Two views of retrieval are reported:
- `retrieval`: the ranked list before abstention, at each k. This isolates ranking
  quality and is comparable to the LegalBench-RAG paper.
- `end_to_end`: the same metrics after abstention (an abstained query scores 0).
Abstention is reported in both directions on the answerable benchmark queries and
the unanswerable set of the same split.

Every run on the test split is appended to experiments/test_runs.jsonl so the
report can state exactly how many times test was used.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from contract_intelligence.config import PipelineConfig, load_config
from contract_intelligence.evaluation import diagnostics as dx
from contract_intelligence.evaluation.metrics import (
    Span,
    first_relevant_rank,
    mean,
    precision,
    precision_merged,
    recall,
    recall_merged,
)
from contract_intelligence.retrieval.retriever import Retriever, decide

K_VALUES = (1, 2, 4, 8, 16, 32, 64)


def load_split(config: PipelineConfig, split: str) -> tuple[list[dict], list[dict]]:
    answerable = json.loads(
        config.resolve(f"data/processed/queries/{split}.json").read_text(encoding="utf-8")
    )
    negatives = [
        n
        for n in json.loads(config.resolve("data/queries/unanswerable.json").read_text(encoding="utf-8"))
        if n["split"] == split
    ]
    return answerable, negatives


def evaluate(config: PipelineConfig, retriever: Retriever, split: str) -> tuple[dict, list[dict]]:
    answerable, negatives = load_split(config, split)
    max_k = max(max(K_VALUES), config.retrieval.top_k)
    diag_k = config.retrieval.top_k

    ranked = retriever.candidates([r["query"] for r in answerable], k=max_k)
    ranked_neg = retriever.candidates([n["query"] for n in negatives], k=1)

    per_query: list[dict] = []
    for record, passages in zip(answerable, ranked):
        gold = [Span(s["file_path"], s["span"][0], s["span"][1]) for s in record["snippets"]]
        spans = [Span(p.file_path, p.start, p.end) for p in passages]
        abstained, _ = decide(passages, config.retrieval)
        top = spans[:diag_k]
        per_query.append({
            "query": record["query"],
            "answerable": True,
            "abstained": abstained,
            "top_score": passages[0].score if passages else None,
            "first_relevant_rank": first_relevant_rank(spans, gold),
            "diagnosis": dx.diagnose_answerable(top, gold, abstained),
            "gold": [list(g) for g in gold],
            "retrieved_top": [[*s, round(p.score, 4)] for s, p in zip(top, passages)],
            "precision": {k: precision(spans[:k], gold) for k in K_VALUES},
            "recall": {k: recall(spans[:k], gold) for k in K_VALUES},
            "precision_merged": {k: precision_merged(spans[:k], gold) for k in K_VALUES},
            "recall_merged": {k: recall_merged(spans[:k], gold) for k in K_VALUES},
            "doc_hit": {k: any(s.file_path in {g.file_path for g in gold} for s in spans[:k]) for k in K_VALUES},
        })
    for record, passages in zip(negatives, ranked_neg):
        abstained, _ = decide(passages, config.retrieval)
        per_query.append({
            "query": record["query"],
            "answerable": False,
            "category": record["category"],
            "abstained": abstained,
            "top_score": passages[0].score if passages else None,
            "diagnosis": dx.diagnose_unanswerable(abstained),
            "retrieved_top": [[p.file_path, p.start, p.end, round(p.score, 4)] for p in passages],
        })

    pos = [q for q in per_query if q["answerable"]]
    neg = [q for q in per_query if not q["answerable"]]

    def at_k(key: str, rows: list[dict], gate: bool = False) -> dict:
        return {
            k: round(mean([0.0 if gate and q["abstained"] else float(q[key][k]) for q in rows]), 4)
            for k in K_VALUES
        }

    answered_rate = mean([0.0 if q["abstained"] else 1.0 for q in pos])
    abstain_rate = mean([1.0 if q["abstained"] else 0.0 for q in neg])
    ranks = [q["first_relevant_rank"] for q in pos]
    summary = {
        "config": config.name,
        "split": split,
        "n_answerable": len(pos),
        "n_unanswerable": len(neg),
        "top_k": config.retrieval.top_k,
        "abstain_threshold": config.retrieval.abstain_threshold,
        "retrieval": {
            "precision": at_k("precision", pos),
            "recall": at_k("recall", pos),
            "precision_merged": at_k("precision_merged", pos),
            "recall_merged": at_k("recall_merged", pos),
            "doc_hit_rate": at_k("doc_hit", pos),
            "mrr": round(mean([1 / r if r else 0.0 for r in ranks]), 4),
        },
        "end_to_end": {
            "precision": at_k("precision", pos, gate=True),
            "recall": at_k("recall", pos, gate=True),
        },
        "abstention": {
            "unanswerable_correctly_abstained": round(abstain_rate, 4),
            "answerable_wrongly_abstained": round(1 - answered_rate, 4),
            "balanced_accuracy": round((answered_rate + abstain_rate) / 2, 4),
            "by_unanswerable_category": {
                cat: f"{sum(q['abstained'] for q in neg if q['category'] == cat)}/"
                f"{sum(q['category'] == cat for q in neg)}"
                for cat in sorted({q["category"] for q in neg})
            },
        },
        f"diagnosis_at_k{diag_k}": dict(Counter(q["diagnosis"] for q in per_query)),
    }
    return summary, per_query


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_run(config: PipelineConfig, summary: dict, per_query: list[dict], out_root: Path) -> Path:
    directory = out_root / f"{config.name}__{summary['split']}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (directory / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a configuration on one split.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--out", default="experiments/runs")
    parser.add_argument("--reason", default="", help="why test is being run (required for test)")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.reason:
        parser.error("--reason is required for test runs; test is for final reporting only")

    config = load_config(args.config)
    started = time.perf_counter()
    summary, per_query = evaluate(config, Retriever.from_config(config), args.split)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    directory = save_run(config, summary, per_query, config.resolve(args.out))

    if args.split == "test":
        log = config.resolve("experiments/test_runs.jsonl")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": config.name,
                "git_commit": _git_commit(),
                "reason": args.reason,
            }) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {directory}")


if __name__ == "__main__":
    main()
