"""Rule-based proposer: map the dominant failure modes to configuration changes.

Each rule names the failure it targets, one config change (dotted key -> value) and
the hypothesis behind it. The proposer ranks untried rules by how many train
failures their target accounts for, so the loop works on the biggest problem
first. Rules were written from the Phase 1-2 error analysis and ablations, all on
train data. No LLM is involved: proposals are deterministic and reviewable, and no
contract text leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    name: str
    targets: tuple[str, ...]  # diagnose() keys whose counts make this rule urgent
    changes: dict = field(hash=False)
    hypothesis: str


RULES = [
    Rule("abstain_when_unrouted", ("false_answer:contract_not_in_corpus",),
         {"graph.abstain_when_unrouted": True},
         "Queries naming an agreement the router cannot match are about contracts outside the "
         "corpus; returning nothing should catch them without touching routed queries."),
    Rule("no_section_title_boost", ("ranked_too_low", "partial_coverage"),
         {"graph.section_title_weight": 0.0},
         "Many extracted 'section titles' are the first words of numbered paragraphs; the Phase 2 "
         "ablation suggests the boost misranks clauses."),
    Rule("no_graph_expansion", ("ranked_too_low",),
         {"graph.expand_top": 0},
         "REFERENCES/DEFINES expansion pulls in referenced sections that rarely hold the answer."),
    Rule("larger_rerank_pool", ("ranked_too_low", "not_retrieved"),
         {"hybrid.rerank_pool": 80},
         "Relevant clauses ranked below 40 never reach the cross-encoder; a larger pool lets it "
         "promote them."),
    Rule("larger_cross_encoder", ("ranked_too_low", "partial_coverage"),
         {"hybrid.reranker_model": "cross-encoder/ms-marco-MiniLM-L12-v2"},
         "A 12-layer cross-encoder should order the routed clauses better than the 6-layer one."),
    Rule("bge_small_embeddings", ("ranked_too_low", "not_retrieved", "wrong_document"),
         {"embedding.model_name": "BAAI/bge-small-en-v1.5",
          "embedding.query_prefix": "Represent this sentence for searching relevant passages: "},
         "A stronger retrieval-tuned embedding model (same size class) should rank the gold clause "
         "higher before reranking."),
    Rule("chunk_size_1000", ("partial_coverage",),
         {"chunking.chunk_size": 1000},
         "Partially covered answers are cut by chunk boundaries; larger sentence chunks should "
         "contain more of each answer."),
    Rule("chunk_size_600", ("partial_coverage", "ranked_too_low"),
         {"chunking.chunk_size": 600},
         "Smaller chunks give the reranker more focused passages and raise precision."),
    Rule("rrf_k_20", ("ranked_too_low",),
         {"hybrid.rrf_k": 20},
         "A smaller RRF constant trusts the top of each list more, favouring clauses that both "
         "the graph ranking and BM25 put near the top."),
    Rule("bm25_b_0.5", ("ranked_too_low",),
         {"hybrid.bm25_b": 0.5},
         "Weaker length normalisation stops BM25 from favouring very short fragments."),
    Rule("graph_pool_100", ("not_retrieved",),
         {"hybrid.graph_pool": 100, "hybrid.bm25_pool": 100},
         "Larger candidate pools recover answers missing from the top 50 of both lists."),
    Rule("tighter_routing", ("wrong_document",),
         {"graph.route_keep_ratio": 0.8},
         "Keeping fewer near-tie contracts should reduce clauses from the wrong agreement."),
]


def propose(diagnosis: dict[str, int], tried: set[str]) -> Rule | None:
    """Untried rule whose targets account for the most failures (ties: list order)."""
    best, best_weight = None, 0
    for rule in RULES:
        if rule.name in tried:
            continue
        weight = sum(diagnosis.get(t, 0) for t in rule.targets)
        if weight > best_weight:
            best, best_weight = rule, weight
    return best
