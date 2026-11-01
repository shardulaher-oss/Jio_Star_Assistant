"""Measure what the LLM answer layer adds, on TRAIN (test only with --reason, logged).

    python -m contract_intelligence.answering.evaluate --config configs/final.yaml

For every query the frozen retriever runs first; the LLM sees its top passages.
- Evidence span: the verified quote is scored as a single retrieved span with the
  same character-level precision/recall as retrieval (0 when abstained, not found or
  ungrounded), and compared with retrieval at k = 1 and 5.
- Abstention: the system abstains when retrieval abstains OR the LLM answers
  "not_found"; compared with retrieval-only abstention.
- Grounding rate, latency of uncached calls, and token usage.
Responses come from the committed cache when present, so re-running is free.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from contract_intelligence.answering.generator import AnswerGenerator
from contract_intelligence.config import load_config
from contract_intelligence.evaluation.metrics import Span, mean, precision, recall
from contract_intelligence.evaluation.runner import load_split
from contract_intelligence.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.split == "test" and not args.reason:
        parser.error("--reason is required for test runs")

    config = load_config(args.config)
    retriever = Retriever.from_config(config)
    generator = AnswerGenerator(config.answer)
    answerable, negatives = load_split(config, args.split)
    cache = {k: v for k, v in generator.cache.items()}

    rows = []
    started = time.perf_counter()
    queries = [(r, True) for r in answerable] + [(n, False) for n in negatives]
    for i, (record, is_answerable) in enumerate(queries, 1):
        result = retriever.search(record["query"])
        answer = generator.answer(result)
        row = {
            "query": record["query"], "answerable": is_answerable, "category": record.get("category"),
            "retrieval_abstained": result.abstained, "answer_type": answer.answer_type,
            "grounded": answer.grounded, "cached": answer.cached, "latency_ms": answer.latency_ms,
            "evidence": [answer.evidence.file_path, answer.evidence.start, answer.evidence.end]
            if answer.evidence else None,
            "answer": answer.answer,
        }
        key = next((k for k in generator.cache if k not in cache), None)
        if key:
            row["usage"] = generator.cache[key].get("_usage")
            cache[key] = generator.cache[key]
        if is_answerable:
            gold = [Span(s["file_path"], *s["span"]) for s in record["snippets"]]
            spans = [Span(p.file_path, p.start, p.end) for p in result.passages]
            ev = [Span(*row["evidence"])] if row["evidence"] else []
            row |= {"evidence_precision": precision(ev, gold), "evidence_recall": recall(ev, gold),
                    "retrieval_p1": precision(spans[:1], gold), "retrieval_r1": recall(spans[:1], gold),
                    "retrieval_p5": precision(spans[:5], gold), "retrieval_r5": recall(spans[:5], gold)}
        rows.append(row)
        print(f"[{i}/{len(queries)}] {answer.answer_type:9} grounded={answer.grounded} "
              f"cached={answer.cached}", flush=True)

    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    called = [r for r in rows if not r["retrieval_abstained"]]
    fresh = [r["latency_ms"] for r in called if not r["cached"]]
    usage = [r["usage"] for r in rows if r.get("usage")]

    def abstains(r: dict, use_llm: bool) -> bool:
        return r["retrieval_abstained"] or (use_llm and r["answer_type"] == "not_found")

    def abstention(use_llm: bool) -> dict:
        correct = mean([abstains(r, use_llm) for r in neg])
        false = mean([abstains(r, use_llm) for r in pos])
        return {"correct_abstention": round(correct, 4), "false_abstention": round(false, 4),
                "balanced_accuracy": round((correct + 1 - false) / 2, 4)}

    summary = {
        "config": config.name, "split": args.split, "model": config.answer.model,
        "n_answerable": len(pos), "n_unanswerable": len(neg),
        "llm_calls": len(called), "uncached_calls_this_run": len(fresh),
        "span_metrics": {
            "llm_evidence_precision": round(mean([r["evidence_precision"] for r in pos]), 4),
            "llm_evidence_recall": round(mean([r["evidence_recall"] for r in pos]), 4),
            "retrieval_p@1": round(mean([r["retrieval_p1"] for r in pos]), 4),
            "retrieval_r@1": round(mean([r["retrieval_r1"] for r in pos]), 4),
            "retrieval_p@5": round(mean([r["retrieval_p5"] for r in pos]), 4),
            "retrieval_r@5": round(mean([r["retrieval_r5"] for r in pos]), 4),
            "evidence_overlaps_gold": round(mean([r["evidence_recall"] > 0 for r in pos]), 4),
        },
        "answer_types_answerable": dict(Counter(r["answer_type"] for r in pos)),
        "answer_types_unanswerable": dict(Counter(r["answer_type"] for r in neg)),
        "grounded_rate_of_answers": round(mean([r["grounded"] for r in called if r["answer_type"] != "not_found"]), 4),
        "abstention_retrieval_only": abstention(False),
        "abstention_retrieval_plus_llm": abstention(True),
        "latency_ms_uncached": {"median": round(statistics.median(fresh), 1), "max": round(max(fresh), 1)}
        if fresh else None,
        "tokens_per_call": {"input_mean": round(mean([u["input_tokens"] or 0 for u in usage])),
                            "output_mean": round(mean([u["output_tokens"] or 0 for u in usage]))}
        if usage else None,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(config.resolve(f"experiments/runs/{config.name}_llm__{args.split}"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.split == "test":
        with config.resolve("experiments/test_runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "config": f"{config.name}+llm", "reason": args.reason}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
