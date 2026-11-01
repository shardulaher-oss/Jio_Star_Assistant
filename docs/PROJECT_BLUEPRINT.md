# Contract Intelligence Assistant: Project Blueprint

This document is an original engineering plan based on the supplied six-page brief. It is not a copy of the confidential assignment document.

## 1. Business problem

A media company signs many content, distribution, sponsorship, and co-production agreements. Legal, business, and content-operations teams repeatedly need answers such as:

- whether a title can be sublicensed to a platform;
- when an exclusivity window ends;
- who owns derivative-work intellectual property.

Today, answering these questions requires manually searching long contracts. That is slow and makes it easy to miss the precise qualifying language, a definition, or a cross-reference. The target capability is not generic chat: it is evidence-first retrieval of the exact clause, with a safe answer of “no sufficiently relevant passage found” when the indexed corpus cannot support an answer.

## 2. Expected application behavior

For a question, the system should:

1. normalize and validate the input;
2. search the indexed contract passages;
3. rank candidates;
4. return a list of passages, each with document identity and original character offsets;
5. abstain when the evidence score is below a configured threshold;
6. make the result inspectable in the web UI and machine-readable in the CLI.

The core deliverable is retrieval quality and honest measurement, not a generated legal opinion. Any explanatory text must remain subordinate to the retrieved evidence and must not imply legal advice.

## 3. Requirements extracted from the brief

### Mandatory assignment requirements

- Use Python as the preferred implementation language unless there is a strong reason not to.
- Use only the CUAD portion of LegalBench-RAG-mini.
- Select exactly 194 questions using the authors' supplied `mini(tests, n=194)` logic, including its per-file seeded random key.
- Index only the contracts referenced by those 194 questions.
- Read documents as UTF-8 text in normal Python text mode and preserve the original document character offsets.
- Validate every annotated answer by checking `document[start:end]` against the supplied answer text before indexing.
- Use the exact split: SHA-256 of `query.strip().encode("utf-8")`, convert the digest to an integer, and assign `test` when the integer modulo 10 is below 3; otherwise assign `train`.
- Create and label at least 30 unanswerable questions. Split them with the same function.
- Commit the complete unanswerable set before beginning the improvement loop, and do not edit it afterward.
- Provide ranked passages with source document and character offsets.
- Abstain when no passage is relevant enough.
- Provide both a web application and a CLI.
- Keep pipeline choices in configuration rather than hard-coding them.
- Implement character-level precision and recall at multiple values of `k` as defined by the LegalBench-RAG paper.
- Measure abstention in both directions: correct abstentions on unanswerable questions and incorrect abstentions on answerable benchmark questions.
- Build a baseline, declare a primary metric and guardrails, and run a single-command improvement loop.
- The loop must diagnose failures, propose and accept/reject changes, log every iteration, and run at least five iterations beyond baseline.
- Use only training data for development and tuning; reserve test for final reporting.
- Include code, README, evaluation report, experiment log, baseline/final configurations, unanswerable data, and core tests.
- Make a fresh clone reproducible: setup, data download, app, CLI, evaluation, loop, and tests must be documented.
- Document assumptions, AI-tool usage, and how to reproduce reported metrics.
- Keep API keys out of the repository and cache any LLM outputs used.

### Optional or stretch requirements

- Use an LLM for query rewriting, reranking, contextualized chunks, or proposing experiments.
- Compare local and hosted LLM choices if an LLM is used, including cost, latency, privacy, and measurable benefit.
- Add hybrid retrieval with a reranker.
- Use clause-aware chunking.
- Resolve definitions and cross-references at indexing time.
- Create train-only sub-splits if the improvement loop needs them.
- Add extra retrieval metrics beyond the required character metrics.

### Recommended but not required

- Sentence Transformers for dense embeddings.
- FAISS for dense nearest-neighbor search.
- BM25 for lexical retrieval.
- Reciprocal Rank Fusion (RRF) to combine lexical and dense rankings.
- A Cross-Encoder for final reranking.
- Streamlit for the web app.
- `argparse` for the CLI.
- YAML configuration files.
- Pytest for unit and integration tests.

### Unnecessary for the core assignment

- A generative LLM is not required: the assignment is primarily retrieval.
- LangChain or LlamaIndex is not required; using them is acceptable only if their internals remain explainable.
- A production database, distributed vector store, Kubernetes, authentication, and cloud deployment are unnecessary for a local benchmark demo.
- Full document summarization or autonomous legal advice would distract from the scored task.

## 4. End-to-end workflow

### Data preparation

1. Download `benchmarks/cuad.json` and `corpus/cuad/`.
2. Load all CUAD tests and apply the exact `mini()` selector.
3. Collect the unique file paths referenced by the selected snippets.
4. Read only those files as UTF-8 text.
5. Validate every gold span and retain its document path, start, end, and answer text.
6. Assign train/test using the exact SHA-256 function.
7. Add at least 30 hand-authored, answerable-by-nothing-in-the-index questions. Label them and freeze the file before experimentation.

### Indexing

1. Parse documents into stable chunks while retaining `start_char` and `end_char`.
2. Store a chunk record containing document path, offsets, text, and optional section metadata.
3. Build a BM25 index over normalized text while keeping the original text for output.
4. Build Sentence Transformer embeddings over the selected chunk text and add them to FAISS.
5. Save indexes plus a manifest containing model names, dimensions, chunking settings, and corpus hashes.

### Query-to-evidence path

1. Receive a question from Streamlit or `argparse`.
2. Optionally normalize the query without changing its meaning.
3. Retrieve lexical and dense candidate lists.
4. Fuse lists with RRF, preserving each candidate's provenance.
5. Optionally rerank the short list with a Cross-Encoder.
6. Apply an abstention policy using score, margin, and/or calibrated train-only thresholds.
7. Return top-k evidence records or an explicit abstention result.
8. For each returned record, verify that the displayed text is exactly `document[start:end]`.

### Evaluation path

Run the same retrieval code without gold spans as inputs. Compare returned spans with gold spans only in the evaluator. Produce required character precision/recall at several k values, document/section diagnostics, and abstention confusion statistics. Save immutable JSON/CSV results and a human-readable report.

## 5. Concepts in simple language

- **Legal document retrieval:** finding the small passage in a long agreement that supports a question, rather than asking a model to remember or summarize the whole file.
- **RAG:** retrieval-augmented generation; retrieve evidence first, then optionally give that evidence to a language model. This project can stop after retrieval.
- **Chunking:** breaking a document into searchable passages. Chunks need enough context to answer a question but must not be so large that matches become imprecise. Offsets must refer to the original unmodified document.
- **Embeddings:** numeric vectors representing semantic meaning. “Can license to a streaming service?” can be close to “right to distribute through digital platforms” even when words differ.
- **BM25:** a lexical ranking method that rewards useful query terms and discounts terms appearing in almost every document. It is strong for exact legal wording and names.
- **FAISS:** a library for fast nearest-neighbor search over embedding vectors.
- **Hybrid retrieval:** using BM25 and dense retrieval together so exact wording and semantic similarity complement each other.
- **RRF:** a simple rank-fusion rule. A candidate receives approximately `1 / (constant + rank)` from each list; candidates high in several lists rise without requiring incomparable scores to be calibrated.
- **Reranking:** a second, more expensive model examines query-candidate pairs and reorders a small candidate set. A Cross-Encoder reads both texts together and usually improves precision.
- **Abstention:** deliberately returning nothing when evidence is weak. It is safer than confidently returning an unrelated clause, especially for questions not answerable from the indexed contracts.

## 6. Proposed system architecture

```text
CUAD JSON + selected corpus files
        |
        v
Data loader -> mini selector -> offset validator -> exact split/frozen negatives
        |
        v
Chunker (offset-preserving records)
        |----------------------|
        v                      v
BM25 index               Sentence Transformer -> FAISS
        |----------------------|
                     v
             Candidate retriever
                     v
                 RRF fusion
                     v
          Optional Cross-Encoder
                     v
             Abstention policy
                |          |
                v          v
          CLI JSON     Streamlit UI
                |
                v
Evaluator -> metrics -> experiment log -> improvement controller
```

Configuration controls chunk size/overlap, normalization, embedding model, retrieval k, RRF constant, reranker, abstention thresholds, and output paths. The index manifest makes an index incompatible with the wrong configuration or corpus obvious.

## 7. Recommended technology stack

| Area | Recommendation | Status | Reason |
|---|---|---|---|
| Language | Python 3.11+ | Preferred/strong recommendation | Fast iteration and good retrieval ecosystem |
| Data and schemas | `json`, `pathlib`, `dataclasses` or Pydantic | Required in spirit | Explicit, testable records and paths |
| Lexical retrieval | `rank_bm25` | Recommended | Small, transparent local index |
| Dense retrieval | `sentence-transformers` | Recommended | Reproducible embedding baseline |
| Vector search | `faiss-cpu` | Recommended | Simple local ANN/flat search |
| Fusion | In-house RRF function | Recommended | Few lines, easy to audit and test |
| Reranking | `CrossEncoder` from Sentence Transformers | Optional/stretch | Useful after a strong baseline |
| Web UI | Streamlit | Recommended | Minimal demo surface |
| CLI | `argparse` | Recommended | Standard library, no extra framework |
| Configuration | PyYAML plus dataclass validation | Recommended | Enables experiment changes without code edits |
| Tests | Pytest | Recommended | Clear unit and end-to-end checks |
| Metrics | NumPy/Pandas only if useful; plain Python is sufficient | Optional | Avoid unnecessary dependencies |
| LLM | Local first; hosted only with explicit justification | Optional | Privacy, cost, latency, and reproducibility |

The minimum viable stack is Python, JSON/path handling, a lexical retriever, an evaluator, Streamlit, `argparse`, and Pytest. Dense retrieval, RRF, reranking, and LLM features should be added only when measurements justify them.

## 8. Proposed modular project structure

```text
.
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   └── final.yaml
├── data/
│   ├── README.md
│   ├── queries/
│   │   └── unanswerable.json
│   └── raw/                         # ignored: downloaded benchmark
├── docs/
│   └── PROJECT_BLUEPRINT.md
├── src/
│   └── contract_intelligence/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── schemas.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── benchmark.py
│       │   ├── splits.py
│       │   └── offsets.py
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── chunking.py
│       │   ├── bm25_index.py
│       │   ├── dense_index.py
│       │   └── manifest.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── lexical.py
│       │   ├── dense.py
│       │   ├── fusion.py
│       │   ├── reranking.py
│       │   └── abstention.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── diagnostics.py
│       │   └── runner.py
│       ├── improvement/
│       │   ├── __init__.py
│       │   ├── diagnose.py
│       │   ├── propose.py
│       │   ├── controller.py
│       │   └── logging.py
│       └── web/
│           ├── __init__.py
│           └── app.py
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── download_data.ps1
│   └── build_index.ps1
├── REPORT.md                         # created after experiments
└── experiments/
    └── iterations.jsonl              # generated, tracked
```

### File and module responsibilities

- `pyproject.toml`: package metadata and pinned/compatible dependencies.
- `configs/*.yaml`: named pipeline configurations; baseline and final must be reproducible.
- `schemas.py`: typed query, span, chunk, result, and metric records.
- `config.py`: YAML loading, validation, and defaults.
- `data/benchmark.py`: dataset loading, exact mini selection, referenced-file discovery.
- `data/splits.py`: the exact SHA-256 split and split counts.
- `data/offsets.py`: UTF-8 reading and span validation.
- `indexing/chunking.py`: offset-preserving chunk creation.
- `indexing/bm25_index.py`: lexical index build/load/search.
- `indexing/dense_index.py`: embedding, FAISS build/load/search.
- `indexing/manifest.py`: corpus/config/model fingerprints and compatibility checks.
- `retrieval/*.py`: independent retrieval stages, RRF, optional reranking, and abstention.
- `evaluation/metrics.py`: character-level metrics at each k and abstention metrics.
- `evaluation/diagnostics.py`: failure categories such as wrong document, wrong section, partial coverage, and bad abstention.
- `evaluation/runner.py`: train/test-safe evaluation orchestration and result persistence.
- `improvement/*.py`: train-only diagnosis, proposal, acceptance gate, and immutable iteration logging.
- `cli.py`: commands for preparation, indexing, querying, evaluation, and improvement.
- `web/app.py`: Streamlit presentation layer calling the same retrieval service as the CLI.
- `tests/`: offset, split, selection, retrieval, abstention, CLI, and UI smoke tests.
- `REPORT.md`: final evaluation narrative, tables, failure analysis, limitations, and next steps.

## 9. Dependency direction

`schemas` and `config` are foundational. `data` depends on them. `indexing` depends on `data` and `schemas`; `retrieval` depends on loaded indexes and `schemas`; `evaluation` depends on retrieval plus gold data but must never feed gold spans into inference; `improvement` depends on evaluation and configuration; `cli` and `web` depend on the public retrieval/evaluation services.

The web layer must not contain retrieval logic. The evaluator must be the only code that reads gold spans during scoring. Index builders must never mutate source documents. This separation prevents accidental leakage and makes CLI/UI behavior identical.

## 10. Main challenges and mitigations

- **Character offsets:** tokenization, Unicode normalization, newline conversion, and trimming can shift indices. Keep the original text untouched, store absolute character boundaries, and test reconstruction with `document[start:end]`.
- **Chunk boundaries:** answer spans may cross a chunk edge. Use overlap and evaluate coverage, while keeping source offsets tied to the original text.
- **Definitions:** a defined term may change meaning throughout the agreement. Preserve surrounding section context and optionally add definition-aware expansion as a measured experiment.
- **Cross-references:** “as defined in Section 4.2” may require retrieving both the referring clause and target section. Treat cross-reference expansion as optional and report whether it helps.
- **Inconsistent formatting:** headings, numbering, OCR artifacts, and whitespace differ across files. Normalize only the index representation; never normalize the source text used for offsets.
- **Unanswerable questions:** a high score is not enough; calibrate abstention on frozen negative questions and track false abstentions on answerable questions.
- **Small benchmark:** changes can overfit 194 queries. Tune only on train, keep test untouched, and report the number of test runs.
- **Reproducibility:** record library/model versions, configuration hashes, corpus hashes, random seeds, hardware, and cached model/LLM outputs.
- **Confidentiality:** keep the provided PDF, secrets, and real contracts out of Git; prefer local models if sensitive data is introduced.

## 11. Phase-by-phase roadmap

### Phase 0 — Repository and contract

- Initialize Git and add `.gitignore`.
- Add this blueprint and a runnable-project README.
- Define the data directory policy and assumptions.

### Phase 1 — Dataset integrity

- Download the two permitted CUAD data locations.
- Implement the exact mini selector and split.
- Validate all spans and produce a query/corpus manifest.
- Author and commit at least 30 unanswerable questions before optimization.

### Phase 2 — Baseline

- Implement simple offset-preserving fixed/paragraph chunking.
- Build BM25 retrieval.
- Implement required character metrics and abstention measurements.
- Record baseline train results and freeze the test evaluator.

### Phase 3 — Dense and hybrid retrieval

- Add Sentence Transformer embeddings and FAISS.
- Add RRF and compare lexical, dense, and hybrid variants.
- Keep all decisions in YAML and add index compatibility checks.

### Phase 4 — Improvement loop

- Add failure diagnosis and a single-command controller.
- Define one primary metric and guardrails before experiments.
- Run at least five post-baseline iterations, accepting or rejecting proposals by the declared criterion.
- Log every result, including failed ideas.

### Phase 5 — Optional quality improvements

- Add Cross-Encoder reranking, clause-aware chunking, definition/cross-reference handling, or a local LLM only as isolated experiments.
- Measure cost and latency as well as retrieval gains.

### Phase 6 — Interfaces and hardening

- Build Streamlit and CLI on the same service.
- Add tests for offsets, split exactness, frozen negatives, persistence, abstention, and representative queries.
- Verify a fresh-clone run from README instructions.

### Phase 7 — Reporting and review

- Generate `REPORT.md` with baseline/final train/test tables, query counts, iteration history, at least 10 concrete failures, limitations, and next steps.
- Review every dependency and explain every design choice in preparation for the debrief.

## 12. Initial assumptions to document

- “Passage” means a returned chunk or span that can be mapped exactly to source character offsets.
- The indexed corpus is limited to the unique contract files referenced by the selected 194 queries.
- The system retrieves evidence; it does not make an independent legal conclusion.
- A question is unanswerable when no indexed passage meets the configured evidence threshold, even if a related concept exists elsewhere.
- The final test set is evaluated only after the configuration is frozen, and every test run is counted.
