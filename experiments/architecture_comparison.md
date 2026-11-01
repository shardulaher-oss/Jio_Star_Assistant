# Architecture comparison (train only)

Four architectures, all independently runnable, on the same corpus (29 contracts),
the same sentence-aware 800-character chunks (1,817), the same embedding model
(all-MiniLM-L6-v2) and the same evaluation code. Train split only: **140 answerable
benchmark queries and 30 unanswerable queries**. Test has not been run. No
architecture choice uses test data.

| Architecture | Config | Pipeline |
|---|---|---|
| Traditional RAG | `configs/traditional_rag.yaml` | dense FAISS on the full query |
| Graph RAG | `configs/graph_rag.yaml` | graph contract routing → question-only dense + section-title boost → REFERENCES/DEFINES expansion |
| Hybrid RAG | `configs/hybrid_rag.yaml` | BM25 + dense (full query) → RRF → cross-encoder on the question (top 40) |
| Graph-enhanced Hybrid RAG | `configs/graph_hybrid_rag.yaml` | graph contract routing → graph ranking + routed BM25 (question) → RRF → cross-encoder on the question (top 40) |

`configs/baseline.yaml` (fixed 500-character chunks) is the Phase 1 reference.
Reproduce the tables with
`python -m contract_intelligence.evaluation.compare baseline traditional_rag graph_rag hybrid_rag graph_hybrid_rag`.

## Retrieval (merged character-level metrics, n = 140)

| Config | R@1 | R@4 | R@8 | R@16 | R@64 | P@1 | P@8 | MRR | Right contract @1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.026 | 0.194 | 0.278 | 0.364 | 0.565 | 0.046 | 0.038 | 0.198 | 0.993 |
| traditional_rag | 0.060 | 0.247 | 0.374 | 0.504 | 0.737 | 0.044 | 0.040 | 0.209 | 0.993 |
| graph_rag | 0.225 | 0.508 | 0.659 | 0.790 | **0.940** | 0.139 | **0.062** | 0.428 | 0.986 |
| hybrid_rag | 0.077 | 0.259 | 0.364 | 0.497 | 0.679 | 0.071 | 0.035 | 0.221 | 0.243 |
| **graph_hybrid_rag** | **0.367** | **0.642** | **0.727** | **0.831** | **0.940** | **0.202** | **0.062** | **0.574** | 0.979 |

## Abstention (each threshold calibrated on train for its own score)

| Config | Score used | Threshold | Unanswerable abstained (n = 30) | Answerable wrongly abstained (n = 140) | Balanced acc. | Absent clause | Not in corpus | Out of scope |
|---|---|---|---|---|---|---|---|---|
| baseline | cosine | 0.6085 | 50.0% | 7.9% | 0.711 | 2/15 | 8/10 | 5/5 |
| traditional_rag | cosine | 0.5719 | 43.3% | 0.7% | 0.713 | 2/15 | 7/10 | 4/5 |
| graph_rag | graph score | 0.5214 | 70.0% | 31.4% | 0.693 | 12/15 | 5/10 | 4/5 |
| hybrid_rag | cross-encoder logit | 1.5354 | 76.7% | 52.1% | 0.623 | 13/15 | 5/10 | 5/5 |
| **graph_hybrid_rag** | cross-encoder logit | −5.6523 | 70.0% | 15.0% | **0.775** | **15/15** | 3/10 | 3/5 |

## Cost (one laptop CPU: Intel, 10 cores, no GPU, 15.7 GB RAM; relative, not absolute)

| Config | Dense index build | Graph build | BM25 build | Startup | Memory added | Median latency | p95 latency |
|---|---|---|---|---|---|---|---|
| traditional_rag | 37.4 s | – | – | 5.2 s | 346 MB | **7.8 ms** | 10.3 ms |
| graph_rag | 39.9 s | 1.5 s | – | 6.1 s | 441 MB | **5.7 ms** | **7.1 ms** |
| hybrid_rag | 39.7 s | – | 0.1 s | 6.0 s | 449 MB | 888 ms | 1,773 ms |
| graph_hybrid_rag | 40.0 s | 1.5 s | 0.07 s | 7.1 s | 542 MB | 571 ms | 1,325 ms |

- The dense index build is the same work for every architecture (embedding 1,817
  chunks). Run-to-run differences of a few seconds are machine load.
- Latency is single-query `search()` over 50 train queries after one warm-up.
  "Memory added" is the resident memory of the constructed retriever: the embedding
  model and index for all four, plus the graph and title embeddings, the BM25
  postings, or the cross-encoder.
- On disk: the FAISS index and chunks take 4.2 MB, the graph 224 KB (gzipped JSON).
  BM25 is rebuilt in memory at startup (under 0.1 s).
- The two reranked architectures were also memory-sensitive on this 16 GB laptop:
  two cross-encoder runs failed with out-of-memory while other processes held most
  of the RAM. They succeeded with a reranking batch of 8 pairs.

## Retrieval complexity per query

N = chunks in the corpus (1,817), n = chunks in the routed contract (about 60 on
average), d = embedding dimension (384), P = rerank pool (40), C = contracts (29).

| Architecture | Work per query |
|---|---|
| Traditional RAG | 1 bi-encoder pass + exact inner product with N vectors: O(N·d) |
| Graph RAG | 1 bi-encoder pass + routing O(C · profile size) + O(n·d) scoring + O(sections) title lookups + edge expansion over the top 5 seeds |
| Hybrid RAG | 1 bi-encoder pass + O(N·d) dense + BM25 over the postings of the query terms + RRF over ≤100 items + **P cross-encoder passes** |
| Graph-Hybrid RAG | Graph RAG's work + BM25 restricted to n chunks + RRF + **P cross-encoder passes** |

The cross-encoder dominates: each pass reads query plus clause (up to 512 tokens)
through a 6-layer transformer, so 40 passes cost about 100× a FAISS search. Graph
RAG is the fastest because it scores only the routed contract's clauses.

## Conclusions

1. **Separating "which contract" from "which clause" is the decisive design choice.**
   Both graph architectures route first. Both non-graph architectures cannot, and
   suffer: Traditional RAG ranks the contract's preamble first, and Hybrid RAG's
   question-only reranker picks clauses from the wrong contract (right contract
   @1 = 24%).
2. **Combining everything is not automatically better.** Hybrid without routing is
   no better than dense. Adding global dense search to the reranked graph-hybrid
   pipeline drops recall@8 from 0.727 to 0.543 (`experiments/runs/gh_*`). The
   reranker only helps when every candidate comes from the routed contract.
3. **Best quality: Graph-Hybrid RAG.** Recall@1 0.367 (6× Traditional RAG),
   recall@8 0.727, MRR 0.574, and the best balanced abstention (0.775, all 15
   absent-clause queries caught). The price is about 0.6 s per query on CPU and
   +200 MB for the cross-encoder.
4. **Best quality per unit cost: Graph RAG.** Recall@8 0.659 and recall@64 0.940 at
   6 ms per query, with no extra model. Its weaker abstention (31% wrong
   abstentions) is the main thing to fix.
5. **Weak spot shared by the graph architectures**: agreements that are not in the
   corpus (3–5/10 caught). When routing finds no contract, they search everything
   and find a plausible clause. Abstaining when the query names an agreement the
   router cannot match is the obvious next step.

These are train results chosen on train. The gap to test is measured once, at the
end, for the final configuration.
