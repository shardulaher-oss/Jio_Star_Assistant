"""Graph RAG: route to the named contract, rank clauses within it, expand via edges.

1. Query processing: split "Consider the <agreement>; <question>" (retrieval.query).
2. Contract routing: IDF-weighted overlap between the query and each contract's
   profile, built from graph nodes (party names, defined aliases in the preamble)
   plus the contract's file name and opening clause.
3. Clause scoring inside the routed contracts: cosine(question, clause) plus
   section_title_weight * cosine(question, title of the clause's section).
4. Expansion from the top seeds along REFERENCES (clause -> referenced section's
   heading clause) and DEFINES (definition of a term the question uses).
"""

from __future__ import annotations

import bisect
import math
import re
from collections import defaultdict

import networkx as nx
import numpy as np

from contract_intelligence.config import GraphConfig
from contract_intelligence.graph.extraction import PREAMBLE_CHARS
from contract_intelligence.indexing.dense_index import DenseIndex, Encoder
from contract_intelligence.retrieval.query import parse_query

_STOP = {
    "the", "and", "for", "of", "to", "in", "on", "a", "an", "by", "between", "consider", "this",
    "that", "with", "agreement", "contract", "inc", "llc", "ltd", "limited", "corporation",
    "corp", "company", "co", "plc", "lp", "is", "are", "be", "any", "under", "what", "does",
}


def tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # split CamelCase file names
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP}


class ContractRouter:
    def __init__(self, profiles: dict[str, set[str]]) -> None:
        self.profiles = profiles
        n = len(profiles)
        df: dict[str, int] = defaultdict(int)
        for profile in profiles.values():
            for token in profile:
                df[token] += 1
        self.idf = {t: math.log(n / c) for t, c in df.items()}

    def scores(self, text: str) -> dict[str, float]:
        query = tokens(text)
        return {fp: sum(self.idf[t] for t in query & profile) for fp, profile in self.profiles.items()}

    def route(self, text: str, min_score: float, keep_ratio: float) -> set[str] | None:
        """Contracts the text refers to, or None when no contract is named clearly."""
        scores = self.scores(text)
        best = max(scores.values(), default=0.0)
        if best < min_score:
            return None
        return {fp for fp, s in scores.items() if s >= keep_ratio * best}


class GraphRanker:
    def __init__(self, encoder: Encoder, dense: DenseIndex, graph: nx.MultiDiGraph, config: GraphConfig) -> None:
        self.encoder = encoder
        self.dense = dense
        self.graph = graph
        self.config = config
        self.vectors = dense.index.reconstruct_n(0, dense.index.ntotal)
        self.position = {f"clause:{c.chunk_id}": i for i, c in enumerate(dense.chunks)}
        self.by_file: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(dense.chunks):
            self.by_file[chunk.file_path].append(i)

        # Section of each clause, and the clause that contains each section heading.
        self.clause_section: dict[int, str] = {}
        sections: list[str] = []
        for clause, position in self.position.items():
            for _, target, data in graph.out_edges(clause, data=True):
                if data["relation"] == "BELONGS_TO" and target.startswith("section:"):
                    self.clause_section[position] = target
        self.heading_clause: dict[str, int] = {}
        for node, data in graph.nodes(data=True):
            if data["type"] == "section":
                sections.append(node)
                positions = self.by_file[data["file_path"]]
                starts = [dense.chunks[p].start for p in positions]
                self.heading_clause[node] = positions[max(0, bisect.bisect_right(starts, data["start"]) - 1)]
        titles = [graph.nodes[s]["title"] for s in sections]
        self.section_row = {s: i for i, s in enumerate(sections)}
        self.title_vectors = encoder.encode_passages(titles) if titles else np.zeros((0, dense.index.d))

        # Definitions: term -> defining clause position, per file.
        self.definitions: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for node, data in graph.nodes(data=True):
            if data["type"] == "definition":
                for source, _, edge in graph.in_edges(node, data=True):
                    if edge["relation"] == "DEFINES" and source in self.position:
                        self.definitions[data["file_path"]].append((data["term"], self.position[source]))

        profiles: dict[str, set[str]] = {}
        for node, data in graph.nodes(data=True):
            if data["type"] != "contract":
                continue
            fp = data["file_path"]
            parts = [fp, dense.chunks[self.by_file[fp][0]].text]
            for neighbour in graph.predecessors(node):
                if graph.nodes[neighbour]["type"] == "party":
                    parts.append(graph.nodes[neighbour]["name"])
            parts += [t for t, p in self.definitions[fp] if dense.chunks[p].start < PREAMBLE_CHARS]
            profiles[fp] = tokens(" ".join(parts))
        self.router = ContractRouter(profiles)

    def _candidates(self, routed: set[str] | None) -> list[int]:
        if routed is None:
            return list(range(len(self.dense.chunks)))
        return [p for fp in sorted(routed) for p in self.by_file[fp]]

    def rank(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        parsed = [parse_query(q) for q in queries]
        texts = [p.question if self.config.question_only and p.question else q for p, q in zip(parsed, queries)]
        query_vectors = self.encoder.encode_queries(texts)
        title_scores = query_vectors @ self.title_vectors.T if len(self.title_vectors) else None

        results = []
        for qi, (query, parse) in enumerate(zip(queries, parsed)):
            routed = None
            if self.config.route_contracts:
                routed = self.router.route(parse.contract or query, self.config.route_min_score,
                                           self.config.route_keep_ratio)
            candidates = self._candidates(routed)
            scores = self.vectors[candidates] @ query_vectors[qi]
            score = {}
            for position, s in zip(candidates, scores):
                section = self.clause_section.get(position)
                if section is not None and title_scores is not None:
                    s += self.config.section_title_weight * title_scores[qi, self.section_row[section]]
                score[position] = float(s)

            # Expansion along graph edges from the best seeds.
            seeds = sorted(score, key=score.get, reverse=True)[: self.config.expand_top]
            question = parse.question.lower()
            for seed in seeds:
                boosted = score[seed] * self.config.expansion_decay
                for _, target, data in self.graph.out_edges(f"clause:{self.dense.chunks[seed].chunk_id}", data=True):
                    if data["relation"] == "REFERENCES" and target in self.heading_clause:
                        neighbour = self.heading_clause[target]
                        if neighbour in score and score[neighbour] < boosted:
                            score[neighbour] = boosted
            if seeds:
                boosted = score[seeds[0]] * self.config.expansion_decay
                files = routed if routed is not None else {self.dense.chunks[s].file_path for s in seeds}
                for fp in files:
                    for term, position in self.definitions[fp]:
                        if len(term) > 3 and term.lower() in question and score.get(position, 1e9) < boosted:
                            score[position] = boosted

            ranked = sorted(score.items(), key=lambda item: item[1], reverse=True)[:k]
            results.append(ranked)
        return results
