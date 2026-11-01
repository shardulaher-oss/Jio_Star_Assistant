"""Print a markdown comparison of saved runs (and benchmarks, when present).

    python -m contract_intelligence.evaluation.compare traditional_rag graph_rag hybrid_rag graph_hybrid_rag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="config names (run folder prefixes)")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--runs", default="experiments/runs")
    parser.add_argument("--benchmarks", default="experiments/benchmarks")
    args = parser.parse_args()

    ks = ("1", "3", "5", "10")
    print(f"Retrieval ({args.split}; LegalBench-RAG character-level metrics, mean over queries)\n")
    print("| Config | n | " + " | ".join(f"P@{k} | R@{k}" for k in ks) + " | R@64 | MRR | Right contract @1 |")
    print("|---" * (2 * len(ks) + 5) + "|")
    for name in args.names:
        s = load(Path(args.runs) / f"{name}__{args.split}" / "summary.json")
        if s is None:
            print(f"| {name} | (no run) |")
            continue
        r = s["retrieval"]
        cells = [f"{r['precision'][k]:.3f} | {r['recall'][k]:.3f}" for k in ks]
        print(f"| {name} | {s['n_answerable']} | " + " | ".join(cells)
              + f" | {r['recall']['64']:.3f} | {r['mrr']:.3f} | {r['doc_hit_rate']['1']:.3f} |")

    print("\nAbstention\n")
    print("| Config | Threshold | n unans. / ans. | Correct abstention | Results returned for unanswerable | False abstention (answerable rejected) | Balanced acc. | Abstained by category |")
    print("|---|---|---|---|---|---|---|---|")
    for name in args.names:
        s = load(Path(args.runs) / f"{name}__{args.split}" / "summary.json")
        if s is None:
            continue
        a = s["abstention"]
        cats = ", ".join(f"{k.split('_')[0]} {v}" for k, v in a["by_unanswerable_category"].items())
        print(f"| {name} | {s['abstain_threshold']} | {s['n_unanswerable']} / {s['n_answerable']}"
              f" | {a['unanswerable_correctly_abstained']:.1%} | {1 - a['unanswerable_correctly_abstained']:.1%}"
              f" | {a['answerable_wrongly_abstained']:.1%} | {a['balanced_accuracy']:.3f} | {cats} |")

    rows = [(n, load(Path(args.benchmarks) / f"{n}.json")) for n in args.names]
    if any(b for _, b in rows):
        print("\nCost (one CPU machine; relative, not absolute)\n")
        print("| Config | Dense index build (s) | Graph build (s) | BM25 build (s) | Startup (s) | Memory added (MB) | Median latency (ms) | p95 latency (ms) |")
        print("|---|---|---|---|---|---|---|---|")
        for name, b in rows:
            if b is None:
                continue
            build = b["build"]
            print(f"| {name} | {build.get('dense_index_seconds', '-')} | {build.get('graph_seconds', '-')}"
                  f" | {build.get('bm25_seconds', '-')} | {b['startup_seconds']} | {b['memory_mb_added_by_retriever']}"
                  f" | {b['latency_ms']['median']} | {b['latency_ms']['p95']} |")


if __name__ == "__main__":
    main()
