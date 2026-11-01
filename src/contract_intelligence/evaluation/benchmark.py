"""Cost measurements for one pipeline: build time, startup, latency and memory.

- build: fresh (uncached) dense index build, plus the graph and BM25 builds when the
  pipeline uses them. The dense index is shared by every Phase 2 architecture.
- startup: time and resident memory to construct the Retriever from cached artifacts.
- latency: wall time of single-query `search()` calls over train queries, as a user
  of the CLI or web app would issue them (no batching).
Numbers are from one CPU machine and vary run to run; treat them as relative.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import psutil

from contract_intelligence.config import load_config
from contract_intelligence.evaluation.runner import load_split


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def benchmark(config_path: str, n_queries: int) -> dict:
    config = load_config(config_path)
    pipeline = config.retrieval.pipeline
    import torch  # noqa: F401  (count library import memory in the baseline, not the pipeline)

    from contract_intelligence.indexing.dense_index import SentenceTransformerEncoder, build_index
    from contract_intelligence.retrieval.retriever import Retriever

    base_rss = rss_mb()
    started = time.perf_counter()
    retriever = Retriever.from_config(config)
    startup = time.perf_counter() - started
    startup_rss = rss_mb()

    build: dict[str, float] = {}
    started = time.perf_counter()
    build_index(config, SentenceTransformerEncoder(config.embedding))
    build["dense_index_seconds"] = round(time.perf_counter() - started, 2)
    if pipeline in ("graph", "graph_hybrid"):
        from contract_intelligence.graph.builder import build_graph

        started = time.perf_counter()
        build_graph(config)
        build["graph_seconds"] = round(time.perf_counter() - started, 2)
    if pipeline in ("bm25", "hybrid", "graph_hybrid"):
        from contract_intelligence.indexing.bm25_index import BM25Index

        started = time.perf_counter()
        BM25Index([c.text for c in retriever.dense.chunks])
        build["bm25_seconds"] = round(time.perf_counter() - started, 2)

    answerable, _ = load_split(config, "train")
    queries = [r["query"] for r in answerable[:n_queries]]
    retriever.search(queries[0])  # warm-up
    latencies = []
    for query in queries:
        started = time.perf_counter()
        retriever.search(query)
        latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()

    return {
        "config": config.name,
        "pipeline": pipeline,
        "reranker": config.hybrid.reranker_model or None if pipeline in ("hybrid", "graph_hybrid") else None,
        "chunks": len(retriever.dense.chunks),
        "build": build,
        "startup_seconds": round(startup, 2),
        "memory_mb_added_by_retriever": round(startup_rss - base_rss),
        "latency_ms": {
            "n": len(latencies),
            "median": round(statistics.median(latencies), 1),
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1),
            "max": round(latencies[-1], 1),
        },
        "machine": f"{platform.processor() or platform.machine()}, {psutil.cpu_count(logical=False)} cores, CPU only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", type=int, default=50, help="train queries to time")
    parser.add_argument("--out", default="experiments/benchmarks")
    args = parser.parse_args()
    result = benchmark(args.config, args.queries)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['config']}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
