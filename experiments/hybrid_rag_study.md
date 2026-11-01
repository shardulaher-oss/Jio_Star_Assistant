# Hybrid RAG: BM25 + dense, RRF and cross-encoder reranking (train only)

Same corpus, chunks (sentence-aware, 800 characters, no overlap), embedding model and
evaluation code as the dense reference `configs/experiments/sentence_800_0.yaml`.
Train split only: 140 answerable and 30 unanswerable queries. Test has not been run.
Variant configs are in `configs/experiments/hybrid/`. The chosen architecture is
`configs/hybrid_rag.yaml`.

Components:
- **BM25**: Okapi BM25 (k1 = 1.5, b = 0.75), implemented in `indexing/bm25_index.py`.
- **Reciprocal Rank Fusion**: `retrieval/fusion.py`, k = 60. Rank-based, so BM25 and
  cosine scores need no common scale.
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L6-v2`, which rescores the top 40
  fused candidates (`retrieval/reranking.py`).
- **Candidate pools**: 50 from FAISS and 50 from BM25, configurable.

## Results (merged metrics, n = 140 answerable)

| Variant | R@1 | R@4 | R@8 | R@16 | R@64 | P@1 | P@8 | MRR | Right contract @1 | Eval time |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense only (reference) | 0.060 | 0.247 | **0.374** | **0.504** | **0.737** | 0.044 | **0.040** | 0.209 | 0.993 | 8 s |
| BM25 only, full query | 0.000 | 0.083 | 0.176 | 0.316 | 0.575 | 0.000 | 0.016 | 0.076 | 1.000 | 45 s |
| BM25 only, question | 0.038 | 0.084 | 0.126 | 0.193 | 0.275 | 0.033 | 0.013 | 0.086 | 0.071 | 41 s |
| RRF (dense + BM25), no reranker | 0.016 | 0.178 | 0.287 | 0.453 | 0.679 | 0.011 | 0.028 | 0.146 | 1.000 | 54 s |
| RRF + reranker on full query | 0.007 | 0.167 | 0.297 | 0.422 | 0.679 | 0.001 | 0.030 | 0.129 | 1.000 | 213 s |
| **RRF + reranker on question** (`hybrid_rag`) | **0.077** | **0.259** | 0.364 | 0.497 | 0.679 | **0.071** | 0.035 | **0.221** | 0.243 | 195–361 s |
| RRF (BM25 on question) + reranker on question | 0.051 | 0.124 | 0.223 | 0.357 | 0.695 | 0.042 | 0.022 | 0.144 | 0.079 | 182 s |
| Dense only + reranker on question | 0.075 | 0.214 | 0.315 | 0.464 | 0.695 | 0.062 | 0.032 | 0.205 | 0.229 | 449 s |

Evaluation time covers 170 queries on CPU and varied with machine load. Latency is
measured separately in the architecture comparison.

## Abstention (`hybrid_rag`, threshold calibrated on train: cross-encoder logit 1.5354)

| Unanswerable abstained | Answerable wrongly abstained | Balanced acc. | Absent clause | Not in corpus | Out of scope |
|---|---|---|---|---|---|
| 76.7% | 52.1% | 0.623 | 13/15 | 5/10 | 5/5 |

## Findings

1. **Hybrid retrieval does not beat dense alone on this benchmark.** Every variant has
   lower recall@8 than dense only (0.374). The best (`hybrid_rag`) is slightly better
   at the very top (R@1 0.077 vs 0.060, MRR 0.221 vs 0.209) and worse below.
2. **Why: nothing separates "which contract" from "which clause".** On the full query,
   BM25 matches party names even more strongly than the embedding does, so fusion
   adds more preamble chunks. The cross-encoder on the full query also favours the
   title chunk. On the question only, the cross-encoder ranks good clauses but from
   the wrong contracts (right contract @1 falls to 24%), because nothing restricts it
   to the named agreement.
3. **R@64 is capped at 0.679** because the fused pool holds at most 100 candidates,
   many of them the same preamble chunks.
4. **The reranker is expensive**: 40 cross-encoder passes per query on CPU, which
   made evaluation 25–50× slower than dense. Its abstention score is no better
   balanced (0.623 vs 0.713 for calibrated dense).
5. **Conclusion**: lexical matching and reranking only help once the search is
   confined to the right contract. That is the motivation for graph-enhanced hybrid
   retrieval, which uses the graph router first.
