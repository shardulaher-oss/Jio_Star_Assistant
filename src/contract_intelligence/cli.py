"""Command-line interface. Uses the same Retriever as the web app and evaluation.

Examples:
    python -m contract_intelligence.cli query "Who owns IP in derivative works?"
    python -m contract_intelligence.cli query "governing law" --top-k 3 --json
    python -m contract_intelligence.cli build-index --config configs/baseline.yaml
    python -m contract_intelligence.cli evaluate --split train
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import asdict, replace

from contract_intelligence.config import PipelineConfig, load_config
from contract_intelligence.schemas import SearchResult


def with_overrides(
    config: PipelineConfig, top_k: int | None = None, threshold: float | None = None,
    no_abstain: bool = False,
) -> PipelineConfig:
    retrieval = config.retrieval
    if top_k is not None:
        retrieval = replace(retrieval, top_k=top_k)
    if threshold is not None:
        retrieval = replace(retrieval, abstain_threshold=threshold)
    if no_abstain:
        retrieval = replace(retrieval, abstain_threshold=None)
    return replace(config, retrieval=retrieval)


def format_result(result: SearchResult, width: int = 100, max_chars: int = 600) -> str:
    lines = [f"Query: {result.query}"]
    if result.abstained:
        lines.append(f"No answer found. {result.message}")
        return "\n".join(lines)
    lines.append(result.message)
    for p in result.passages:
        lines.append("")
        lines.append(f"#{p.rank}  score {p.score:.3f}  {p.file_path}  [chars {p.start}-{p.end}]")
        text = " ".join(p.text.split())
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        lines.extend(textwrap.wrap(text, width=width, initial_indent="    ", subsequent_indent="    "))
    return "\n".join(lines)


def cmd_query(args: argparse.Namespace) -> int:
    from contract_intelligence.retrieval.retriever import Retriever

    config = with_overrides(load_config(args.config), args.top_k, args.threshold, args.no_abstain)
    result = Retriever.from_config(config).search(args.question)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False) if args.json else format_result(result))
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from contract_intelligence.indexing.dense_index import (
        SentenceTransformerEncoder,
        index_dir,
        load_or_build,
    )

    config = load_config(args.config)
    dense = load_or_build(config, SentenceTransformerEncoder(config.embedding), rebuild=args.rebuild)
    print(json.dumps({**dense.manifest, "path": str(index_dir(config))}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from contract_intelligence.evaluation import runner

    argv = ["--config", args.config, "--split", args.split]
    if args.reason:
        argv += ["--reason", args.reason]
    runner.main(argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contract-search", description="Clause search over the indexed contracts."
    )
    parser.add_argument("--config", default="configs/baseline.yaml", help="pipeline YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    query = sub.add_parser("query", help="search for the clauses that answer a question")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, help="override retrieval.top_k")
    query.add_argument("--threshold", type=float, help="override retrieval.abstain_threshold")
    query.add_argument("--no-abstain", action="store_true", help="always return passages")
    query.add_argument("--json", action="store_true", help="print machine-readable JSON")
    query.set_defaults(func=cmd_query)

    build = sub.add_parser("build-index", help="embed chunks and build the FAISS index")
    build.add_argument("--rebuild", action="store_true")
    build.set_defaults(func=cmd_build_index)

    evaluate = sub.add_parser("evaluate", help="evaluate the config on a split")
    evaluate.add_argument("--split", choices=("train", "test"), default="train")
    evaluate.add_argument("--reason", default="", help="required for test runs")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
