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
import sys
import textwrap
from dataclasses import asdict, replace

from contract_intelligence.config import ConfigError, PipelineConfig, load_config
from contract_intelligence.logging_utils import setup_logging
from contract_intelligence.schemas import SearchResult


def top_k_value(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return value


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


def shorten(text: str, max_chars: int | None) -> str:
    """Shorten for display at the last sentence end before max_chars, never mid-sentence."""
    if not max_chars or len(text) <= max_chars:
        return text
    cut = max(text.rfind(". ", 0, max_chars), text.rfind("; ", 0, max_chars))
    return (text[: cut + 1] if cut > 0 else text[:max_chars]) + " [...]"


def format_result(result: SearchResult, width: int = 100, max_chars: int | None = None) -> str:
    lines = [f"Query: {result.query}"]
    if result.abstained:
        lines.append(f"No answer found. {result.message}")
        return "\n".join(lines)
    lines.append(result.message)
    for p in result.passages:
        lines.append("")
        lines.append(f"#{p.rank}  score {p.score:.3f}  {p.file_path}  [chars {p.start}-{p.end}]")
        text = shorten(" ".join(p.text.split()), max_chars)
        lines.extend(textwrap.wrap(text, width=width, initial_indent="    ", subsequent_indent="    "))
    return "\n".join(lines)


def cmd_query(args: argparse.Namespace) -> int:
    from contract_intelligence.retrieval.retriever import Retriever

    config = with_overrides(load_config(args.config), args.top_k, args.threshold, args.no_abstain)
    result = Retriever.from_config(config).search(args.question)
    answer = None
    if args.answer:
        from contract_intelligence.answering.generator import AnswerGenerator

        try:
            answer = AnswerGenerator(config.answer).answer(result)
        except Exception as error:  # retrieval results are still useful without the answer
            print(f"warning: answer generation failed ({type(error).__name__}: {error}); "
                  "showing retrieval results only.", file=sys.stderr)
    if args.json:
        payload = asdict(result) | ({"answer": answer.to_dict()} if answer else {})
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if answer:
            print(format_answer(answer) + "\n")
        print(format_result(result, max_chars=args.max_chars))
    return 0


def format_answer(answer, width: int = 100) -> str:
    lines = [f"ANSWER ({answer.answer_type}, confidence {answer.confidence}, model {answer.model})"]
    lines += textwrap.wrap(answer.answer, width=width, initial_indent="    ", subsequent_indent="    ")
    if answer.evidence:
        e = answer.evidence
        lines.append(f"  Evidence: passage #{e.passage_rank}  {e.file_path}  [chars {e.start}-{e.end}]")
        lines += textwrap.wrap(f'"{" ".join(e.text.split())}"', width=width,
                               initial_indent="    ", subsequent_indent="    ")
    if answer.note:
        lines.append(f"  Note: {answer.note}")
    return "\n".join(lines)


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
    parser.add_argument("--config", default="configs/final.yaml", help="pipeline YAML")
    parser.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-json", action="store_true", help="structured JSON logs on stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    query = sub.add_parser("query", help="search for the clauses that answer a question")
    query.add_argument("question")
    query.add_argument("--top-k", type=top_k_value, help="override retrieval.top_k (1-100)")
    query.add_argument("--threshold", type=float, help="override retrieval.abstain_threshold")
    query.add_argument("--no-abstain", action="store_true", help="always return passages")
    query.add_argument("--json", action="store_true", help="print machine-readable JSON")
    query.add_argument("--max-chars", type=int, help="shorten displayed passages at a sentence end")
    query.add_argument("--answer", action="store_true",
                       help="also generate a grounded answer with the configured LLM (Gemini)")
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
    setup_logging(args.log_level, args.log_json)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, PermissionError) as error:
        # Expected operator errors: a clear one-line message, no traceback.
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
