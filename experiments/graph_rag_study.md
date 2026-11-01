# Graph RAG: design, results and ablation (train only)

Config: `configs/graph_rag.yaml` (`retrieval.pipeline: graph`), on the same corpus,
chunks (sentence-aware, 800 characters, no overlap), embedding model and evaluation
code as the dense reference `configs/experiments/sentence_800_0.yaml`. Train split
only: 140 answerable and 30 unanswerable queries. Test has not been run.

## How it works

1. **Query processing.** Split "Consider the <agreement> between <parties>; <question>"
   into the contract reference and the question (`retrieval/query.py`).
2. **Contract routing.** Score each contract by IDF-weighted token overlap between the
   query's contract reference and the contract's profile. The profile is built from
   graph nodes (party names, aliases defined in the preamble) plus the file name and
   opening clause. Search only the contracts scoring at least half the best score,
   when the best score is at least 3.0 (one distinctive token). Otherwise search
   every contract.
3. **Clause scoring.** cosine(question, clause) + 0.3 × cosine(question, title of
   the clause's section, from the BELONGS_TO edge).
4. **Expansion.** From the top 5 clauses, follow REFERENCES edges to the referenced
   section's heading clause. From the top clause, follow DEFINES to the definitions
   of terms the question uses. Neighbours score 0.9 × their seed.

## Results (merged metrics, n = 140 answerable)

| Pipeline | R@1 | R@4 | R@8 | R@16 | R@64 | P@1 | P@8 | MRR | Right contract @1 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (fixed 500, dense) | 0.026 | 0.194 | 0.278 | 0.364 | 0.565 | 0.046 | 0.038 | 0.198 | 0.993 |
| Dense, sentence 800 | 0.060 | 0.247 | 0.374 | 0.504 | 0.737 | 0.044 | 0.040 | 0.209 | 0.993 |
| **Graph RAG** | **0.225** | **0.508** | **0.659** | **0.790** | **0.940** | **0.139** | **0.062** | **0.428** | 0.986 |

Diagnosis at k = 10 (answerable): right contract but wrong section falls from 60
(dense, sentence 800) to 23, and full coverage rises from 42 to 54.

## Abstention (threshold recalibrated on train for each pipeline)

| Pipeline | Threshold | Unanswerable abstained | Answerable wrongly abstained | Balanced acc. | Absent clause | Not in corpus | Out of scope |
|---|---|---|---|---|---|---|---|
| Dense, sentence 800 | 0.6085* | 46.7% | 10.7% | 0.680 | 3/15 | 7/10 | 4/5 |
| Graph RAG | 0.5214 | 70.0% | 31.4% | 0.693 | 12/15 | 5/10 | 4/5 |

\* The baseline's threshold. Calibrating this config separately gives a balanced
accuracy of 0.713.

The graph score measures how well a clause matches the **question**, so it catches
absent clauses (12/15 vs 3/15). The cost is more wrong abstentions on answerable
queries. Agreements that are not in the corpus are caught less often (5/10): routing
falls back to searching every contract and finds a generic clause that matches the
question. An obvious next step: when the query names an agreement but routing
matches none, abstain.

## Ablation: which component helps?

Each row switches off one component of `graph_rag.yaml`
(`configs/experiments/graph_ablation/`).

| Variant | R@1 | R@4 | R@8 | R@16 | R@64 | P@8 | MRR | Right contract @1 |
|---|---|---|---|---|---|---|---|---|
| Full Graph RAG | 0.225 | 0.508 | 0.659 | 0.790 | 0.940 | 0.062 | 0.428 | 0.986 |
| Routing only (full query, no title boost, no expansion) | 0.060 | 0.302 | 0.496 | 0.657 | 0.941 | 0.049 | 0.246 | 1.000 |
| No question-only query (embed the full query) | 0.108 | 0.350 | 0.539 | 0.697 | 0.931 | 0.054 | 0.302 | 1.000 |
| No contract routing | 0.023 | 0.073 | 0.121 | 0.167 | 0.383 | 0.013 | 0.074 | 0.071 |
| No section-title boost | **0.236** | **0.538** | 0.665 | **0.798** | 0.933 | **0.063** | **0.454** | 0.986 |
| No edge expansion | 0.225 | 0.498 | **0.667** | **0.807** | 0.938 | 0.062 | 0.429 | 0.986 |

## Findings

1. **Routing plus question-only embedding is what works.** Routing restricts the
   search to the right contract (from party and alias nodes). With the contract
   fixed, the question alone ranks clauses much better than the full query,
   because the title prefix no longer dominates the embedding. Neither works alone:
   question-only without routing loses the contract (right contract @1 drops to 7%).
2. **The richer graph signals do not help on train.** The section-title boost
   slightly hurts R@1, R@4 and MRR. Many "section titles" are the first words of a
   numbered paragraph, not real headings. Reference and definition expansion is
   neutral to slightly negative: few gold spans sit in a referenced section or a
   definition.
3. **The graph is cheap.** It builds in 1.5 s and loads in 0.1 s. At query time,
   routing is a dictionary lookup and scoring is one matrix product over the routed
   contract's clauses (about 60 on average).

`graph_rag.yaml` keeps the full design, so the architecture comparison is faithful
to it. Turning off the title boost and expansion is left to the improvement loop,
which decides on train evidence.
