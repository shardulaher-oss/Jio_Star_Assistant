"""Categorise retrieval failures of a saved TRAIN run, with concrete examples.

    python -m contract_intelligence.evaluation.error_analysis --config configs/graph_hybrid_rag.yaml

Answerable queries are judged at k = 5 (the loop's primary cut-off). A query can
carry several labels, e.g. an incomplete span caused by a chunk boundary.

  wrong_document          no top-5 passage comes from a gold contract
  wrong_section           gold contract retrieved, but no gold text in the top 64
  ranked_too_low          gold text first appears at rank 6-64
  incomplete_span         gold text in the top 5, but less than 99% of it
  chunk_boundary          ...and the gold span does not fit inside any single chunk
  definition_not_retrieved   the gold span lies in a definition the graph extracted
  cross_reference         the gold span cites a section the graph could not resolve,
                          or sits in a section that a top-5 clause REFERENCES
  semantic_mismatch       the question and the gold text share no content-word stem
                          (5-letter prefix), i.e. the answer uses different vocabulary
  incorrect_abstention    an answerable query returned nothing
  unanswerable_retrieved  an unanswerable query returned passages
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from contract_intelligence.config import load_config
from contract_intelligence.data.corpus import load_documents
from contract_intelligence.graph.builder import load_or_build_graph
from contract_intelligence.graph.extraction import extract_references
from contract_intelligence.indexing.bm25_index import tokenize
from contract_intelligence.indexing.chunking import chunk_corpus
from contract_intelligence.retrieval.query import parse_query

K = 5
_GENERIC = {"contract", "agreement", "clause", "provision", "party", "parties", "shall", "will",
            "may", "include", "includes", "provided", "provide", "terms", "term", "under"}

SUGGESTIONS = {
    "wrong_document": "Improve routing: index party aliases from the whole contract and abstain or "
                      "ask for the agreement when two contracts tie.",
    "wrong_section": "Recall problem before reranking: add the clause's section heading to its "
                     "embedded text, or enlarge the candidate pool.",
    "ranked_too_low": "The answer is retrieved but outranked: a stronger reranker or a larger "
                      "rerank pool should promote it.",
    "incomplete_span": "Return the matched sentence plus its neighbours, or merge adjacent "
                       "retrieved chunks, so long clauses are covered.",
    "chunk_boundary": "Chunk on clause and section boundaries rather than a fixed character "
                      "budget, or add overlap only at boundaries.",
    "definition_not_retrieved": "Index each definition as its own passage and expand queries "
                                "with defined terms they mention.",
    "cross_reference": "Resolve 'Section X' at indexing time, attaching the referenced text to "
                       "the clause; improve heading detection so more targets resolve.",
    "semantic_mismatch": "Query expansion with CUAD category vocabulary (e.g. 'anti-assignment' "
                         "-> 'assign, transfer, consent') or an LLM query rewriter.",
    "incorrect_abstention": "Calibrate per question type, or abstain on the contract-level "
                            "signal (routing) rather than a single clause score.",
    "unanswerable_retrieved": "Add an explicit check that the retrieved clause addresses the "
                              "asked topic (e.g. a trained relevance classifier) before answering.",
}


def content_words(text: str) -> set[str]:
    """Content-word stems (first 5 letters), so 'governing'/'governed' match."""
    return {t[:5] for t in tokenize(text) if len(t) > 3 and t not in _GENERIC}


def excerpt(text: str, start: int, end: int, limit: int = 220) -> str:
    snippet = " ".join(text[start:end].split())
    return snippet if len(snippet) <= limit else snippet[:limit] + " [...]"


def analyse(config_path: str, run_dir: Path) -> tuple[dict, list[dict]]:
    config = load_config(config_path)
    docs = {d.file_path: d.text for d in load_documents(
        config.resolve(config.data.document_manifest), config.resolve(config.data.corpus_root))}
    chunks = chunk_corpus([type("D", (), {"file_path": f, "text": t})() for f, t in docs.items()], config.chunking)
    by_file = defaultdict(list)
    for c in chunks:
        by_file[c.file_path].append(c)
    graph = load_or_build_graph(config)
    definitions = defaultdict(list)
    sections = defaultdict(dict)
    for node, d in graph.nodes(data=True):
        if d["type"] == "definition":
            definitions[d["file_path"]].append((d["start"], d["end"], d["term"]))
        elif d["type"] == "section":
            sections[d["file_path"]][d["number"]] = (d["start"], node)

    rows = [json.loads(l) for l in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    labelled = []
    for q in rows:
        labels = []
        if not q["answerable"]:
            if not q["abstained"]:
                labels.append("unanswerable_retrieved")
            labelled.append({**q, "labels": labels})
            continue
        gold = q["gold"]
        top = q["retrieved_top"][:K]
        rank = q["first_relevant_rank"]
        recall = q["recall"]["5"] if "5" in q["recall"] else q["recall"][5]
        if q["abstained"]:
            labels.append("incorrect_abstention")
        if not {r[0] for r in top} & {g[0] for g in gold}:
            labels.append("wrong_document")
        elif rank is None:
            labels.append("wrong_section")
        elif rank > K:
            labels.append("ranked_too_low")
        elif recall < 0.99:
            labels.append("incomplete_span")
            if not any(any(c.start <= g[1] and g[2] <= c.end for c in by_file[g[0]]) for g in gold):
                labels.append("chunk_boundary")
        if recall < 0.99:
            for fp, start, end in gold:
                if any(a <= start < b or a < end <= b for a, b, _ in definitions[fp]):
                    labels.append("definition_not_retrieved")
                    break
            for fp, start, end in gold:
                cited = [r.key for r in extract_references(docs[fp][start:end])]
                if any(n not in sections[fp] for n in cited):
                    labels.append("cross_reference")
                    break
                top_clauses = {f"clause:{c.chunk_id}" for c in by_file[fp]
                               if any(r[0] == fp and r[1] == c.start for r in top)}
                referenced = {v for u in top_clauses for _, v, e in graph.out_edges(u, data=True)
                              if e["relation"] == "REFERENCES"}
                preceding = [(s, node) for s, node in sections[fp].values() if s <= start]
                gold_section = max(preceding)[1] if preceding else None  # nearest heading above
                if gold_section in referenced:
                    labels.append("cross_reference")
                    break
            question = parse_query(q["query"]).question
            gold_text = " ".join(docs[fp][s:e] for fp, s, e in gold)
            if not content_words(question) & content_words(gold_text):
                labels.append("semantic_mismatch")
        labelled.append({**q, "labels": labels})

    counts = Counter(label for q in labelled for label in q["labels"])
    return {"counts": dict(counts), "docs": docs}, labelled


def render(config_name: str, run_dir: Path, result: dict, labelled: list[dict], per_category: int) -> str:
    docs = result["docs"]
    pos = [q for q in labelled if q["answerable"]]
    neg = [q for q in labelled if not q["answerable"]]
    lines = [
        f"# Error analysis: `{config_name}` (train)", "",
        f"Source run: `{run_dir.as_posix()}`. Answerable queries judged at k = {K}; "
        f"{len(pos)} answerable and {len(neg)} unanswerable train queries. A query can have several labels. "
        "Generated by `python -m contract_intelligence.evaluation.error_analysis`.", "",
        "| Category | Queries | Suggested improvement |", "|---|---|---|",
    ]
    for label, suggestion in SUGGESTIONS.items():
        lines.append(f"| {label} | {result['counts'].get(label, 0)} | {suggestion} |")
    lines += ["", "## Examples", ""]
    n = 0
    for label in SUGGESTIONS:
        examples = [q for q in labelled if label in q["labels"]][:per_category]
        if not examples:
            continue
        lines += [f"### {label}", ""]
        for q in examples:
            n += 1
            lines.append(f"**{n}. {q['query']}**")
            lines.append("")
            lines.append(f"- Labels: {', '.join(q['labels'])}; top score {q['top_score']}; "
                         f"first relevant rank {q.get('first_relevant_rank')}; abstained {q['abstained']}")
            for fp, s, e in q.get("gold", [])[:1]:
                lines.append(f"- Gold `{fp[5:60]}` [{s}:{e}]: \"{excerpt(docs[fp], s, e)}\"")
            if q["retrieved_top"]:
                fp, s, e = q["retrieved_top"][0][:3]
                lines.append(f"- Top-1 `{fp[5:60]}` [{s}:{e}]: \"{excerpt(docs[fp], s, e)}\"")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/graph_hybrid_rag.yaml")
    parser.add_argument("--run", help="run folder (default: experiments/runs/<name>__train)")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--out", default="experiments/error_analysis.md")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run or f"experiments/runs/{config.name}__train")
    if "__test" in run_dir.as_posix() or "test" == run_dir.name.split("__")[-1]:
        parser.error("error analysis for tuning is train-only")
    result, labelled = analyse(args.config, run_dir)
    Path(args.out).write_text(render(config.name, run_dir, result, labelled, args.per_category), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
