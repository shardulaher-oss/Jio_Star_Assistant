"""Build, persist and load the contract knowledge graph (NetworkX MultiDiGraph).

Node types: contract, section, clause, definition, party, date, right, restriction,
ip, owner_mention. Clause nodes are the retrieval chunks, so graph retrieval returns
the same (file, start, end) passages that evaluation scores.

Edges (all carry `relation` and the evidence span `ev_start`/`ev_end`):
  section  BELONGS_TO  contract | parent section
  clause   BELONGS_TO  section (nearest preceding heading) | contract
  party    PARTY_TO    contract              (party named in the preamble)
  clause   DEFINES     definition            (clause contains the definition)
  clause   DEPENDS_ON  definition            (clause uses the defined term)
  clause   REFERENCES  section               (resolved "Section 4.2" style reference)
  clause   GRANTS_RIGHT right                (grant of a licence or right)
  clause   RESTRICTS   restriction           (shall not / may not / prior written consent)
  clause   EXPIRES_ON  date                  (date in a sentence with an expiry cue)
  ip       BELONGS_TO  clause;  ip OWNED_BY definition | party | owner_mention
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import time
from collections import Counter
from pathlib import Path

import networkx as nx

from contract_intelligence.config import PipelineConfig, load_config
from contract_intelligence.data.corpus import load_documents
from contract_intelligence.graph.extraction import Extract, extract_all
from contract_intelligence.indexing.chunking import chunk_document
from contract_intelligence.indexing.dense_index import index_fingerprint
from contract_intelligence.schemas import Chunk, Document

EXTRACTOR_VERSION = "1"


def contract_title(file_path: str) -> str:
    name = Path(file_path).stem
    return re.split(r"-EX-|_EX-", name)[-1].replace("_", " ").strip(" -") or name


def _add_edge(graph: nx.MultiDiGraph, src: str, dst: str, relation: str, ev: tuple[int, int]) -> None:
    graph.add_edge(src, dst, relation=relation, ev_start=ev[0], ev_end=ev[1])


def add_document(graph: nx.MultiDiGraph, document: Document, chunks: list[Chunk]) -> dict:
    fp = document.file_path
    contract = f"contract:{fp}"
    graph.add_node(contract, type="contract", file_path=fp, title=contract_title(fp),
                   start=0, end=len(document.text))

    starts = [c.start for c in chunks]

    def clause_at(offset: int) -> str:
        i = max(0, bisect.bisect_right(starts, offset) - 1)
        return f"clause:{chunks[i].chunk_id}"

    for chunk in chunks:
        graph.add_node(f"clause:{chunk.chunk_id}", type="clause", file_path=fp,
                       start=chunk.start, end=chunk.end, chunk_index=chunk.chunk_index)

    extracts = extract_all(document.text)
    by_kind: dict[str, list[Extract]] = {}
    for e in extracts:
        by_kind.setdefault(e.kind, []).append(e)

    # Sections: keep the first heading per number; parent = dotted prefix if present.
    sections: dict[str, Extract] = {}
    for e in by_kind.get("section", []):
        sections.setdefault(e.key, e)
    for number, e in sections.items():
        node = f"section:{fp}:{number}"
        graph.add_node(node, type="section", file_path=fp, number=number,
                       title=dict(e.attrs)["title"], start=e.start, end=e.end)
        parent = number.rsplit(".", 1)[0] if "." in number else None
        target = f"section:{fp}:{parent}" if parent in sections else contract
        _add_edge(graph, node, target, "BELONGS_TO", (e.start, e.end))
    heading_starts = sorted((e.start, f"section:{fp}:{n}") for n, e in sections.items())
    for chunk in chunks:
        i = bisect.bisect_right([s for s, _ in heading_starts], chunk.start) - 1
        target = heading_starts[i][1] if i >= 0 else contract
        _add_edge(graph, f"clause:{chunk.chunk_id}", target, "BELONGS_TO", (chunk.start, chunk.end))

    for e in by_kind.get("party", []):
        node = f"party:{fp}:{e.key}"
        graph.add_node(node, type="party", file_path=fp, name=e.key, start=e.start, end=e.end)
        _add_edge(graph, node, contract, "PARTY_TO", (e.start, e.end))

    definitions = {e.key: e for e in by_kind.get("definition", [])}
    for term, e in definitions.items():
        node = f"definition:{fp}:{term}"
        graph.add_node(node, type="definition", file_path=fp, term=term,
                       style=dict(e.attrs)["style"], start=e.start, end=e.end)
        _add_edge(graph, clause_at(e.start), node, "DEFINES", (e.start, e.end))
    for term, e in definitions.items():
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])")
        defining = clause_at(e.start)
        for chunk in chunks:
            clause = f"clause:{chunk.chunk_id}"
            match = pattern.search(chunk.text)
            if match and clause != defining:
                ev = (chunk.start + match.start(), chunk.start + match.end())
                _add_edge(graph, clause, f"definition:{fp}:{term}", "DEPENDS_ON", ev)

    resolved = unresolved = 0
    for e in by_kind.get("reference", []):
        if e.key in sections:
            _add_edge(graph, clause_at(e.start), f"section:{fp}:{e.key}", "REFERENCES", (e.start, e.end))
            resolved += 1
        else:
            unresolved += 1

    for e in by_kind.get("date", []):
        node = f"date:{fp}:{e.key}"
        graph.add_node(node, type="date", file_path=fp, value=e.key, start=e.start, end=e.end)
        _add_edge(graph, clause_at(e.start), node, "EXPIRES_ON", (e.start, e.end))
    for e in by_kind.get("right", []):
        node = f"right:{fp}:{e.start}"
        graph.add_node(node, type="right", file_path=fp, kind=e.key,
                       flags=dict(e.attrs)["flags"], start=e.start, end=e.end)
        _add_edge(graph, clause_at(e.start), node, "GRANTS_RIGHT", (e.start, e.end))
    for e in by_kind.get("restriction", []):
        node = f"restriction:{fp}:{e.start}"
        graph.add_node(node, type="restriction", file_path=fp, topic=e.key, start=e.start, end=e.end)
        _add_edge(graph, clause_at(e.start), node, "RESTRICTS", (e.start, e.end))

    party_names = {e.key: f"party:{fp}:{e.key}" for e in by_kind.get("party", [])}
    for e in by_kind.get("ip", []):
        node = f"ip:{fp}:{e.start}"
        graph.add_node(node, type="ip", file_path=fp, start=e.start, end=e.end)
        _add_edge(graph, node, clause_at(e.start), "BELONGS_TO", (e.start, e.end))
        owner = e.key
        if owner in definitions:
            target = f"definition:{fp}:{owner}"
        elif any(owner in name for name in party_names):
            target = next(node_id for name, node_id in party_names.items() if owner in name)
        else:  # keep the literal owner text rather than guessing who it refers to
            target = f"owner_mention:{fp}:{owner}"
            graph.add_node(target, type="owner_mention", file_path=fp, name=owner,
                           start=e.start, end=e.end)
        _add_edge(graph, node, target, "OWNED_BY", (e.start, e.end))

    return {"references_resolved": resolved, "references_unresolved": unresolved}


def build_graph(config: PipelineConfig) -> tuple[nx.MultiDiGraph, dict]:
    documents = load_documents(
        config.resolve(config.data.document_manifest), config.resolve(config.data.corpus_root)
    )
    graph = nx.MultiDiGraph(fingerprint=graph_fingerprint(config))
    totals: Counter = Counter()
    for document in documents:
        totals.update(add_document(graph, document, chunk_document(document, config.chunking)))
    return graph, dict(totals)


def graph_fingerprint(config: PipelineConfig) -> str:
    return f"{index_fingerprint(config)}-x{EXTRACTOR_VERSION}"


def graph_path(config: PipelineConfig) -> Path:
    return config.resolve(config.graph.output_root) / f"{graph_fingerprint(config)}.json.gz"


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph, edges="edges")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(data, handle)


def load_graph(path: Path) -> nx.MultiDiGraph:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return nx.node_link_graph(data, edges="edges", directed=True, multigraph=True)


def load_or_build_graph(config: PipelineConfig, rebuild: bool = False) -> nx.MultiDiGraph:
    path = graph_path(config)
    if path.exists() and not rebuild:
        return load_graph(path)
    graph, _ = build_graph(config)
    save_graph(graph, path)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the contract knowledge graph.")
    parser.add_argument("--config", default="configs/experiments/sentence_800_0.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    started = time.perf_counter()
    graph, totals = build_graph(config)
    build_seconds = time.perf_counter() - started
    path = graph_path(config)
    save_graph(graph, path)
    started = time.perf_counter()
    load_graph(path)
    print(json.dumps({
        "config": config.name,
        "path": str(path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "nodes_by_type": dict(Counter(d["type"] for _, d in graph.nodes(data=True)).most_common()),
        "edges_by_relation": dict(Counter(d["relation"] for *_, d in graph.edges(data=True)).most_common()),
        **totals,
        "build_seconds": round(build_seconds, 1),
        "load_seconds": round(time.perf_counter() - started, 2),
        "file_kb": round(path.stat().st_size / 1024),
    }, indent=2))


if __name__ == "__main__":
    main()
