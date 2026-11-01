# Evaluation Report: Contract Clause Search

## Summary

I built a clause search system over the 29 CUAD contracts in LegalBench-RAG-mini. It
returns ranked passages with exact character offsets, or nothing when no passage is
relevant enough. The final pipeline (`configs/final.yaml`) works in three stages:

1. Identify the contract the question names, using a knowledge graph of parties and
   defined aliases.
2. Rank that contract's clauses by fusing a question-only embedding ranking with BM25.
3. Rerank the top 40 with a cross-encoder, then abstain if the best score is below
   a threshold calibrated on train.

**On the held-out test split (54 answerable queries, evaluated once):**
- Recall@5 is **0.684** (95% CI 0.56–0.80) against **0.087** for the Phase 1
  baseline. Precision@5 is 0.098, recall@1 0.394 and MRR 0.602.
- On the 11 test unanswerable queries it correctly returns nothing for 54.5%. It
  wrongly returns nothing for 7.4% of answerable test queries.

Train and test ranking metrics agree within noise (train recall@5 0.660). Abstention
is the weak part: it caught 15/15 absent-clause train queries but only 2/5 on test.

## System design

**Data and offsets.** The 194 queries are selected with the authors' `mini()` and split
with the given SHA-256 function (140 train / 54 test). Every gold span is checked
against `document[start:end]`, and every contract against a SHA-256 manifest. Chunk
text is never modified: whitespace is collapsed only on the copy sent to models, so
every returned offset points into the original file.

**Key decisions and the alternatives tried** (all on train):

| Decision | Chosen | Alternatives measured | Evidence |
|---|---|---|---|
| Chunking | Sentence/paragraph-aware, 800 characters, no overlap | Fixed 500 (baseline), fixed 500 + 100 overlap, sentence 500 ± 150 overlap, 600, 800 + 200, 1000 | Fixed windows started on a gold boundary for 2/192 gold spans, sentence chunks for 85/192. Sentence 800 raised dense R@8 from 0.278 to 0.374. Overlap never helped: repeated sentences crowd the top-k (`experiments/chunking_study.md`) |
| Contract identification | Route via graph party/alias nodes, file name and opening clause (IDF overlap) | None (search all) | The query prefix "Consider the <agreement> between <parties>" dominates the embedding. Dense search finds the right contract at rank 1 for 99% of queries, but ranks its title page first for 74/140. Routing + question-only scoring: R@8 0.374 → 0.659 (`experiments/graph_rag_study.md`) |
| Lexical signal | BM25 on the question, **restricted to the routed contract** | BM25 over the whole corpus (full query / question) | Unrestricted BM25 matched party names and pulled in preamble chunks: hybrid R@8 0.287 vs 0.374 dense |
| Fusion | Reciprocal Rank Fusion (k = 60) | Score averaging (not tried: BM25 and cosine scales differ) | RRF needs no score calibration |
| Reranker | `ms-marco-MiniLM-L6-v2` on the top 40 routed candidates, question only | No reranker; full-query reranking; adding global dense candidates; L12 model | Reranking inside the routed contract: R@1 0.240 → 0.367. With global candidates it **hurt** (R@8 0.543), because it promotes clauses from other contracts |
| Abstention | Threshold on the top-1 score of each pipeline, calibrated on train by balanced accuracy | Shared threshold; "abstain if the named contract is unknown" | Score scales differ per pipeline (cosine, graph score, logit), so no threshold transfers. None is a probability |
| Graph store | NetworkX MultiDiGraph, gzipped JSON (224 KB) | Neo4j | 3,920 nodes and 9,608 edges fit in memory and build in 1.5 s; a server adds nothing at this scale |

**The knowledge graph** is extracted with rules, not an LLM. It contains:
- **Nodes:** sections, clauses (the retrieval chunks), 679 definitions, parties, expiry
  dates, rights, restrictions and IP-ownership statements.
- **Edges:** BELONGS_TO, DEFINES, DEPENDS_ON, REFERENCES (584 of 903 "Section X"
  references resolved), GRANTS_RIGHT, RESTRICTS, EXPIRES_ON, OWNED_BY and PARTY_TO.
- **Provenance:** every node and edge carries the offsets of the text that produced
  it. Nothing is inferred beyond a pattern match.

The ablation was honest about the graph. **Only routing mattered**: the section-title
boost and the REFERENCES/DEFINES expansion did not improve train metrics, and the loop
rejected them again later.

**Four architectures** are kept runnable (`configs/{traditional,graph,hybrid,graph_hybrid}_rag.yaml`):

| Train, n = 140 | P@1 | R@1 | P@5 | R@5 | R@10 | MRR | Balanced abstention | Median latency | Memory |
|---|---|---|---|---|---|---|---|---|---|
| Traditional RAG (dense) | 0.044 | 0.060 | 0.046 | 0.293 | 0.442 | 0.209 | 0.713 | 7.8 ms | 346 MB |
| Graph RAG | 0.139 | 0.225 | 0.077 | 0.546 | 0.678 | 0.428 | 0.693 | **5.7 ms** | 441 MB |
| Hybrid RAG (BM25 + dense + reranker) | 0.071 | 0.077 | 0.045 | 0.282 | 0.411 | 0.221 | 0.623 | 888 ms | 449 MB |
| **Graph-Hybrid RAG** (= final) | **0.202** | **0.367** | **0.088** | **0.660** | **0.749** | **0.574** | **0.775** | 571 ms | 542 MB |

Index build is about 40 s for all four (embedding 1,817 chunks on CPU), plus 1.5 s for
the graph. Cost details and per-query complexity are in
`experiments/architecture_comparison.md`. If latency mattered more than the last
points of recall, Graph RAG would be the choice: 100× faster with 83% of the R@5.

## LLM use

**No LLM is used** anywhere: not for retrieval, extraction, reranking or the
improvement loop.

- **It wasn't needed for the main problem.** The dominant failure was structural
  (which contract vs. which clause), and routing plus a small cross-encoder fixed it.
- **Rules keep provenance and determinism.** Rule-based graph extraction and
  rule-based proposals are reproducible and every output points to its evidence.
  LLM extraction would need caching and spot-checking for invented relationships.
- **Privacy.** On real JioStar agreements, a hosted LLM (e.g. Gemini) would send
  confidential contract text to a third party. Everything here runs locally on CPU
  with two ~90 MB open models, so no contract text leaves the machine.

Where an LLM could plausibly help, as future work measured against this baseline:
- query rewriting for vocabulary mismatches (e.g. "right of first refusal" vs. an
  option-grant clause);
- a local model (e.g. Qwen via Ollama) as a second-stage relevance check for
  abstention.

## Evaluation setup

- **Metrics.** Character-level precision and recall at k exactly as in the
  LegalBench-RAG authors' code: overlap summed per retrieved snippet, 0 for an empty
  retrieval, and a mean over queries with equal weight. All final pipelines use
  non-overlapping chunks, so these equal the merged-span variants also computed.
  P@k/R@k are measured on the ranking before abstention; end-to-end variants (where
  abstention scores 0) are saved too. MRR and "right contract @1" are diagnostics.
- **Abstention, both directions.**
  - *Correct abstention:* an unanswerable query returns nothing.
  - *Results returned for unanswerable:* the opposite.
  - *False abstention (answerable query rejected):* an answerable query returns nothing.
  - *Balanced accuracy:* the mean of the answer rate on answerable queries and the
    correct-abstention rate.
- **Sample sizes.**
  - Train: 140 answerable queries (192 gold spans) and 30 unanswerable.
  - Test: 54 answerable queries and 11 unanswerable.
  - The unanswerable set has 41 queries: 20 clauses verifiably absent from an indexed
    contract, 13 agreements not in the corpus (3 near misses reusing a real party or
    title), and 8 out-of-scope questions. It was written and committed (`ac3b8b4`)
    before any tuning and never edited.
- **Splits.** All development, ablations, threshold calibration and loop decisions
  used train only. The loop code refuses test runs.
- **Test runs: 2, one per configuration**, both after `configs/final.yaml` was frozen
  in commit `4c7b377`: the final config, and the Phase 1 baseline for the
  baseline-vs-final table. Both are logged in `experiments/test_runs.jsonl`. No test
  result was used to change anything.
- Confidence intervals are 95% bootstrap intervals over queries (5,000 resamples).

## Results

**Baseline vs. final, train and test** (P@k/R@k as defined above):

| Config | Split | n (ans. / unans.) | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | P@10 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | train | 140 / 30 | 0.046 | 0.026 | 0.051 | 0.147 | 0.051 | 0.231 | 0.034 | 0.308 | 0.198 |
| **Final** | train | 140 / 30 | 0.202 | 0.367 | 0.135 | 0.600 | 0.088 | 0.660 | 0.052 | 0.749 | 0.574 |
| Baseline | test | 54 / 11 | 0.018 | 0.006 | 0.022 | 0.064 | 0.019 | 0.087 | 0.023 | 0.198 | 0.122 |
| **Final** | **test** | 54 / 11 | **0.200** | **0.394** | **0.132** | **0.604** | **0.098** | **0.684** | **0.060** | **0.769** | **0.602** |

| Config | Split | Correct abstention | Results returned for unanswerable | False abstention | Balanced acc. | Absent clause / not in corpus / out of scope abstained |
|---|---|---|---|---|---|---|
| Baseline | train | 50.0% | 50.0% | 7.9% | 0.711 | 2/15, 8/10, 5/5 |
| Final | train | 70.0% | 30.0% | 15.0% | 0.775 | 15/15, 3/10, 3/5 |
| Baseline | test | 54.5% | 45.5% | 11.1% | 0.717 | 1/5, 2/3, 3/3 |
| **Final** | **test** | **54.5%** | **45.5%** | **7.4%** | **0.736** | 2/5, 2/3, 2/3 |

95% bootstrap intervals for the final config:

| Metric | Train | Test |
|---|---|---|
| R@5 | 0.660 [0.59, 0.73] | 0.684 [0.56, 0.80] |
| P@5 | 0.088 [0.07, 0.10] | 0.098 [0.07, 0.13] |
| R@1 | 0.367 [0.30, 0.44] | 0.394 [0.27, 0.52] |
| Correct abstention | 0.70 [0.53, 0.87] | 0.55 [0.27, 0.82] |
| False abstention | 0.15 [0.09, 0.21] | 0.07 [0.02, 0.15] |

**Train–test gap.**
- **Ranking generalises.** Test R@5 is slightly *higher* than train (0.684 vs 0.660),
  well inside both intervals. The final pipeline has few tuned numbers (routing
  thresholds, pool sizes), and the loop rejected every tuned change, so there was
  little room to overfit ranking to train.
- **Abstention does not clearly generalise.** Correct abstention fell from 70% to
  55%, and absent-clause catches from 15/15 to 2/5. With 11 test negatives the
  interval is very wide (27–82%), so this is weak evidence either way. Still, the
  threshold was fitted on only 30 train negatives, and the absent-clause cases
  depend on a fine score margin. I treat abstention as **not validated**.
- **Precision is low by construction.** A gold span has a median of about 315
  characters (mean 395) and a chunk averages about 660, so even a perfect top-1 chunk
  scores P@1 ≈ 0.5–0.6. Recall at a
  small k is the more meaningful figure for a reviewer.
- **The baseline's test scores are much lower than its train scores** (R@5 0.087 vs
  0.231). Its failure mode (ranking the title page first) varies by contract, and
  the 54 test queries cover 21 of the 29 contracts. This is one reason the test set
  alone is a noisy comparison.

## Iteration history

**Objective, declared before the loop ran** (committed in `6d12d67`, before the first
iteration):
- **Primary metric: Recall@5.** A reviewer realistically reads about five passages;
  recall says whether the answer text is in front of them.
- **Guardrails:**
  - Precision@5 may drop by at most 0.005, so the loop cannot win just by returning
    longer passages.
  - Balanced abstention accuracy may drop by at most 0.02, because abstention is the
    second required capability.
  - False abstention must stay at or below 20%; refusing answerable questions is
    costly for users.
- **Acceptance:** (A) R@5 gains at least 0.01 (about 1.4 queries) with paired-bootstrap
  P(gain > 0) ≥ 0.90 and all guardrails hold; or (B) R@5 is non-inferior (drop ≤ 0.005)
  while balanced abstention improves by at least 0.02. The bootstrap requirement
  stops a gain from a couple of lucky queries being accepted on n = 140.

**How the loop works** (`python -m contract_intelligence.improvement.loop`):
1. Evaluate the current config on train.
2. Diagnose failures at k = 5: wrong document, ranked too low, partial coverage, not
   retrieved, false answers by category.
3. The rule-based proposer picks the untried rule whose target failure accounts for
   the most queries.
4. Run the candidate on train, recalibrating its threshold on train.
5. Accept or reject, log everything to `experiments/improvement_log.{jsonl,md}`, and
   save the best config to `configs/final.yaml`.

Gold spans are used only to score rankings, never as retriever input.

**Manual exploration before the loop** (Phases 1–2, proposed by me with Claude Code,
evaluated on train; full tables in `experiments/*_study.md`):

| Step | Change | Train R@8 | Train R@5 | Kept? |
|---|---|---|---|---|
| Baseline | fixed 500, dense | 0.278 | 0.231 | reference |
| Chunking | sentence-aware 800, no overlap (5 variants tried) | 0.374 | 0.293 | yes |
| Overlap | 100–200 characters (3 variants) | ≤ 0.372 | – | no |
| Graph RAG | routing + question-only + title boost + expansion (5 ablations) | 0.659 | 0.546 | yes |
| Hybrid | BM25 + dense + RRF (± reranker, 6 variants) | ≤ 0.364 | 0.282 | no (worse than dense) |
| Graph-Hybrid | routing + routed BM25 + reranker (4 variants) | 0.727 | 0.660 | yes → loop start |

**Automated loop** (11 iterations after the start, all proposed by `improvement/propose.py`):

| # | Change | R@1 | R@5 | P@5 | Bal. abst. | False abst. | ΔR@5 | P(gain>0) | Decision |
|---|---|---|---|---|---|---|---|---|---|
| 0 | reference: Phase 1 baseline | 0.026 | 0.231 | 0.051 | 0.711 | 7.9% | | | reference |
| 1 | start: graph_hybrid_rag | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | | | start |
| 2 | section-title boost off | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | +0.000 | 0.00 | rejected |
| 3 | cross-encoder L6 → L12 | 0.406 | 0.684 | 0.089 | 0.780 | 10.7% | +0.024 | 0.87 | rejected (P < 0.90) |
| 4 | chunk size 600 | 0.396 | 0.630 | 0.113 | 0.788 | 15.7% | −0.031 | 0.08 | rejected |
| 5 | BGE-small embeddings | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | −0.001 | 0.00 | rejected |
| 6 | rerank pool 40 → 80 | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | +0.000 | 0.00 | rejected |
| 7 | graph expansion off | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | +0.000 | 0.00 | rejected |
| 8 | RRF k 60 → 20 | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | +0.000 | 0.00 | rejected |
| 9 | BM25 b 0.75 → 0.5 | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | +0.000 | 0.00 | rejected |
| 10 | chunk size 1000 | 0.389 | 0.678 | 0.074 | 0.787 | 9.3% | +0.018 | 0.73 | rejected (P@5 guardrail) |
| 11 | abstain when named contract unknown | 0.367 | 0.660 | 0.088 | 0.781 | 17.1% | +0.000 | 0.00 | rejected (abstention +0.006 < 0.02) |
| 12 | graph and BM25 pools 50 → 100 | 0.367 | 0.660 | 0.088 | 0.775 | 15.0% | +0.000 | 0.00 | rejected |

**What the loop taught.**
1. **Upstream changes no longer matter.** The routed contract has about 60 chunks and
   the cross-encoder already scores the top 40, so any change before reranking
   (title boost, expansion, embeddings, RRF k, BM25 b, pools) leaves the top 5
   unchanged. The remaining gains must come from the reranker or the chunking.
2. **The criteria rejected two plausible gains.** The L12 reranker (+0.024 R@5,
   R@1 0.367 → 0.406, false abstention 15% → 11%) missed the 0.90 bootstrap bar.
   Chunk size 1000 traded 0.014 of P@5 for R@5. Both look promising, but not
   provable on 140 queries, so they stay out.
3. **Abstaining on an unknown contract** caught one more not-in-corpus query (Echo
   Valley) but also rejected the three answerable PC Quote queries whose routing
   misses (false abstention 15% → 17%). Routing is not
   reliable enough to be an abstention signal on its own (see failure A1 below).

## Error analysis

Fourteen concrete train failures of the final configuration, grouped by cause. Labels,
counts and excerpts are generated by `python -m contract_intelligence.evaluation.error_analysis`
(`experiments/error_analysis.md`). Counts over 140 answerable and 30 unanswerable
train queries at k = 5:

| Category | Queries |
|---|---|
| Ranked too low (gold at rank 6–64) | 33 |
| Semantic mismatch | 21 |
| Incorrect abstention | 21 |
| Incomplete span | 20 |
| Cross-reference involved | 11 |
| Unanswerable retrieved | 9 |
| Definition not retrieved | 7 |
| Chunk boundary | 5 |
| Wrong document | 3 |
| Wrong section | 3 |

**A. Routing failures (wrong document)**

1. *PC Quote / A.B. Watley, governing law.* The top-1 passage is King
   Pharmaceuticals' New York governing-law clause. The gold Illinois clause is never
   retrieved. **Diagnosis:** the contract writes the parties as "PCQ" and
   "A.B. Wately" (a typo in the filing), so none of the query's name tokens ("pc",
   "quote", "watley") are in the contract profile. Routing falls back to the whole
   corpus, where any governing-law clause matches. **Fix:** fuzzy and initialism
   matching of party names (edit distance, "PC Quote" → "PCQ"); when routing is
   ambiguous, ask the user to pick the contract.
2. *Same contract, "What licenses are granted?"* The top-1 passage comes from Premier
   Biomedical. Same cause and same fix.

**B. Vocabulary mismatch between the question type and the clause**

3. *Depomed/King, "exclusivity agreement?"* The gold span is "Depomed … hereby grants
   to King an exclusive option…". The top-1 is the entire-agreement clause.
   **Diagnosis:** CUAD's "exclusivity" label covers exclusive options and grants,
   while the question suggests a non-compete style clause. The MS MARCO cross-encoder
   was trained on web QA, not on CUAD's label definitions. **Fix:** expand each
   question type with its CUAD definition vocabulary, or fine-tune the reranker on
   CUAD.
4. *BicycleTX, "right of first refusal / offer / negotiation?"* The gold is a share
   option grant ("you will be granted an option… to acquire 107,417 ordinary
   shares"). It is ranked 27th, and the top score (−8.3) falls below the threshold,
   so the system abstains. **Diagnosis:** the annotators read an option grant as a
   ROFR-type right, which shares no vocabulary with the question. **Fix:** the same
   query expansion. Arguably, abstaining here is defensible behaviour.
5. *Bon-Ton agency agreement, "terminated for convenience?"* The gold is an
   e-commerce wind-down obligation. **Diagnosis:** this looks like an annotation the
   system cannot reasonably match, since no termination wording appears in it. It is
   counted as a failure, but no retrieval change should be tuned towards it.

**C. Right contract, relevant clause ranked too low**

6. *BicycleTX, "exceptions to competitive restrictions?"* The gold paragraph 13.2
   exception (holding up to 3% of listed shares) is at rank 12. The top-1 is the
   paragraph 14.4 exception ("nothing in this paragraph 14 shall prevent you…").
   **Diagnosis:** the top-1 is a valid exception too, but the benchmark annotates
   only one, so the metric understates the system here. **Fix:** none for retrieval.
   In the report this is a limitation of single-span gold labels.
7. *Theravance, "terminated for convenience?"* The gold is "The Company may, in its
   sole and absolute discretion, terminate… with immediate effect" (payment in
   lieu), at rank 8. The top-1 is clause 18, "Termination without notice", which is
   termination for cause. **Diagnosis:** the reranker matches "terminate…
   immediate effect" but not the convenience/cause distinction. **Fix:** a
   legal-domain reranker (the L12 model helped on train but was not proven) or CUAD
   fine-tuning.

**D. Incomplete span and chunk boundary**

8. *Vitamin Shoppe, expiration date.* The 605-character gold sentence crosses a chunk
   boundary, and the piece holding it is at rank 4, so it is only partly covered.
   **Diagnosis:** a long single-sentence clause, and the 800-character budget packs
   whole sentences, so the boundary falls inside the answer region. **Fix:**
   chunk-boundary-aware expansion: also return the neighbouring chunk when a
   retrieved chunk ends mid-clause.
9. *Vitamin Shoppe, cap on liability.* The top-1 chunk starts 271 characters before
   the gold (an SEC redaction notice and the heading "14. LIMITATION OF LIABILITY")
   and ends 183 characters before the gold does. **Diagnosis:** the right clause,
   cut by packing. The clause also excepts "SECTIONS 13(a) AND 13(b)", whose text is
   not attached. **Fix:** section-aware chunking (cut at "14." headings) and
   cross-reference attachment at indexing time.

**E. Incorrect abstention**

10. *HPIL, "anti-assignment clause?"* The top-1 chunk **contains** the gold sentence
    ("…shall not be assignable by any Party"), but the system abstains: the
    cross-encoder logit is −6.35 against a threshold of −5.65. **Diagnosis:** the
    reranker scores the whole 700-character chunk, which is mostly
    successors-and-assigns boilerplate, so the relevant sentence is diluted. One
    global threshold cannot separate this from true negatives. **Fix:** score at the
    sentence level (the maximum over sentences in the chunk), or calibrate
    thresholds per question type.
11. *BicycleTX ROFR* (case 4) abstains for a related reason: no clause matches the
    question's vocabulary, so the best score is low.

**F. Unanswerable queries answered**

12. *"Starlight Pictures / Nimbus Streaming content license: sublicensing?"* (not in
    the corpus). Routing **confidently** chose the Glu Mobile amendment and returned
    its preamble. **Diagnosis:** the router's profile includes file-name words, and
    "content" and "license" from "Content License Agreement" scored above the
    routing threshold, so generic title words were treated like party names.
    **Fix:** route on party and alias tokens only, and require at least one matched
    party name.
13. *"Harbor Lane Films / Meridian Studios co-production: who owns derivative works?"*
    (not in the corpus). Routing found nothing, so the search covered the whole
    corpus and returned Theravance's IP clause above the threshold. **Diagnosis:**
    the topic exists in other contracts, so the clause score is high. Iteration 11
    (abstain when the named contract is unknown) targets exactly this, but it also
    rejected the three answerable PC Quote queries whose routing misses (case 1).
    **Fix:** better routing first, then abstain on routing failure.
14. *Absent clauses on test (3 of 5 answered).* The fixed threshold that caught all
    15 absent-clause train queries let three test ones through. **Diagnosis:** the
    margin between "clause about something nearby" and "clause that answers" is
    small, and the threshold was fitted on 30 negatives. **Fix:** a larger,
    stratified unanswerable set for calibration, and a relevance classifier trained
    on CUAD labels (each CUAD contract also annotates which categories are absent).

## Limitations and next steps

**What doesn't work yet**
- **Abstention is not validated.** Test correct-abstention is 54.5% with a 27–82%
  interval, and absent clauses are caught inconsistently. A legal user cannot yet
  treat "no answer found" as "the contract has no such clause".
- **Routing depends on the query naming the contract.** Benchmark queries always do.
  Free-form questions ("Does this agreement allow sublicensing?") search all 29
  contracts, where the title-page problem returns. A real interface should select
  the contract explicitly (a dropdown or document upload), which is also the safer
  UX.
- **Single-span gold labels understate the system** when a contract has several
  valid clauses (case 6). Precision is capped by chunk length: a perfect chunk scores
  about 0.5.
- **Small samples.** 54 test queries and 11 test negatives give wide intervals.
  Differences under about 0.05 R@5 between configurations are not meaningful here.
- **Cost.** About 0.6 s per query on CPU because of the cross-encoder, and +200 MB of
  memory. The reranker crashed with out-of-memory errors on this 16 GB laptop until
  its batch size was reduced.
- **Graph extraction is heuristic.** 319 of 903 cross-references are unresolved, and
  some "section titles" are paragraph openings. A few party names are generic
  phrases ("Limited Liability Company").

**Before legal users relied on it**
1. Make the contract an explicit input, not something inferred from the question.
2. Rebuild abstention around a validated signal: sentence-level relevance scores,
   CUAD's explicit "category absent" annotations as negatives, and calibration on a
   few hundred negatives with per-category thresholds. Report its precision with
   confidence intervals.
3. Always show the passage in context with its offsets (the web app already does),
   and attach referenced sections and definitions to each passage, so a lawyer can
   verify rather than trust.
4. Re-test the two promising rejected changes (L12 reranker, 1000-character chunks)
   on a larger train set, and try a CUAD-fine-tuned reranker.
5. Evaluate on the full CUAD benchmark (4,000+ queries), not mini, before trusting
   differences of a few points.
6. Keep the system local. If an LLM is added (query rewriting, relevance checking),
   use a self-hosted model, so confidential agreements never leave JioStar's
   infrastructure.
