# Contract Intelligence Assistant

## Overview

A clause search system for commercial contracts. Given a natural-language question,
it returns a ranked list of contract passages, each with its source document and
character offsets into the original file, or returns nothing (and says so) when no
passage is relevant enough. It is evaluated on the CUAD portion of
LegalBench-RAG-mini (194 queries, 29 contracts) plus 41 hand-written unanswerable
queries. It has a Streamlit web app and a CLI, both on the same retrieval backend.

Four retrieval architectures are implemented and selectable by configuration:
traditional dense RAG, Graph RAG, hybrid BM25 + dense RAG with reranking, and
graph-enhanced hybrid RAG. A train-only improvement loop tunes the chosen pipeline.
Results and analysis are in [REPORT.md](REPORT.md).

## Setup

- **Python 3.11** (FAISS and PyTorch wheels are most reliable on 3.11 for Windows).
- Tested on Windows 11, CPU only, 16 GB RAM. No GPU needed.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"               # every command below assumes this (bash: export PYTHONPATH=src)
```

**Data.** Download the benchmark file, then let the preparation script fetch only the
29 contracts the selected queries refer to (from the
[awinml/legalbench-rag](https://huggingface.co/datasets/awinml/legalbench-rag) mirror):

```powershell
New-Item -ItemType Directory -Force data/raw/benchmarks
Invoke-WebRequest https://huggingface.co/datasets/awinml/legalbench-rag/resolve/main/benchmarks/cuad.json -OutFile data/raw/benchmarks/cuad.json
python -m contract_intelligence.data.prepare
```

Expected output: `query_count 194, train_count 140, test_count 54,
referenced_contract_count 29, unanswerable_count 41 (30 train / 11 test)`. The script
fails if any annotated span does not match `document[start:end]`.

**Models.** Downloaded automatically from Hugging Face on first use:
`sentence-transformers/all-MiniLM-L6-v2` (embeddings, about 90 MB) and
`cross-encoder/ms-marco-MiniLM-L6-v2` (reranker, about 90 MB). Retrieval needs no
API keys.

**Optional LLM answers (Gemini).** Only for the answer layer (`--answer`, or the app
toggle). Put your key in a git-ignored `.env` file in the project root. Never commit
it or paste it anywhere:

```powershell
Set-Content -Path .env -Value "GEMINI_API_KEY=<your key>" -Encoding utf8
git check-ignore .env        # must print ".env"
```

The top 5 retrieved passages are sent to Google's API. That is fine for public CUAD;
for confidential contracts set `answer.allow_external: false` (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#5-security-and-privacy)). The cached
answers in `experiments/llm_cache/` reproduce the reported LLM numbers without a key.

## How to run

```powershell
# Web app (open http://localhost:8501; choose the config in the sidebar)
.\.venv\Scripts\streamlit.exe run src/contract_intelligence/web/app.py

# CLI: query (text or --json), build an index, evaluate
python -m contract_intelligence.cli --config configs/final.yaml query "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; What is the governing law for this contract?"
python -m contract_intelligence.cli --config configs/final.yaml query "Who owns the IP in derivative works?" --top-k 3 --json
python -m contract_intelligence.cli query "<question>" --answer      # + grounded Gemini answer with a verified quote
python -m contract_intelligence.cli --log-level INFO --log-json query "<question>"   # structured logs (no query text)
python -m contract_intelligence.cli --config configs/final.yaml build-index
python -m contract_intelligence.cli --config configs/final.yaml evaluate --split train

# Improvement loop (train only; resumes from experiments/improvement_log.jsonl)
python -m contract_intelligence.improvement.loop --iterations 6

# Error analysis, architecture comparison, cost benchmark
python -m contract_intelligence.evaluation.error_analysis --config configs/graph_hybrid_rag.yaml
python -m contract_intelligence.evaluation.compare baseline traditional_rag graph_rag hybrid_rag graph_hybrid_rag
python -m contract_intelligence.evaluation.benchmark --config configs/graph_rag.yaml

# LLM answer layer: evaluate on train (uses the committed cache; no key needed)
python -m contract_intelligence.answering.evaluate --config configs/final.yaml

# Tests: 171 unit + 7 integration (~20 s); unit only: pytest -m "not integration"
pytest
```

`--config` goes before the CLI subcommand. Test-split evaluation requires
`--split test --reason "..."` and is logged to `experiments/test_runs.jsonl`.

## Architecture

Full component, configuration, performance, reliability and security notes:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).


```
data/raw/benchmarks/cuad.json ─► data/prepare.py ─► data/processed/ (194 queries, split, manifests)
data/raw/corpus/cuad/*.txt ────► data/corpus.py (SHA-256 + length checks)
                                    │
                    indexing/segmentation.py + chunking.py   (sentence-aware chunks; offsets kept)
                     │                 │                     │
     indexing/dense_index.py   indexing/bm25_index.py   graph/extraction.py + builder.py
     (MiniLM + FAISS IP)       (Okapi BM25)             (NetworkX graph with provenance)
                     └─────────────────┼─────────────────────┘
                     retrieval/{query, graph_ranker, lexical, fusion, reranking,
                                hybrid_ranker, graph_hybrid_ranker}.py
                                       │
                      retrieval/retriever.py (one backend, threshold abstention)
                     ┌─────────────────┼──────────────────────────┐
                 cli.py          web/app.py      evaluation/{runner, metrics, diagnostics,
                                                 error_analysis, compare, benchmark}.py
                                                          │
                                         improvement/{diagnose, propose, loop}.py
```

- **Configuration** lives in `configs/*.yaml` (loaded and validated by `config.py`):
  chunking, embedding model, retrieval pipeline (`dense | graph | bm25 | hybrid |
  graph_hybrid`), graph routing and expansion, BM25/RRF/pool/reranker settings, top-k
  and the abstention threshold. `configs/baseline.yaml` is the Phase 1 baseline,
  `configs/{traditional,graph,hybrid,graph_hybrid}_rag.yaml` are the four
  architectures, and `configs/final.yaml` is the configuration frozen by the loop.
  `configs/experiments/` holds study variants.
- **Artifacts** (git-ignored, rebuilt automatically): FAISS indexes under
  `artifacts/indexes/<fingerprint>/`, graphs under `artifacts/graphs/`, and
  cross-encoder scores cached under `artifacts/cache/`. Fingerprints cover the corpus
  hashes, chunking and model, so a stale index is never reused.
- **Results** (tracked): `experiments/runs/<config>__<split>/` (summary and per-query
  results), `experiments/loop/` (every loop candidate's config and run),
  `experiments/improvement_log.{jsonl,md}`, `experiments/benchmarks/`, the studies in
  `experiments/*.md`, and `experiments/test_runs.jsonl`.

## Reproducing results

Times are for a 10-core laptop CPU.

| Step | Command | Time |
|---|---|---|
| Prepare data | `python -m contract_intelligence.data.prepare` | 1–2 min (downloads 29 contracts) |
| Train metrics for each architecture | `python -m contract_intelligence.cli --config configs/<name>.yaml evaluate --split train` | dense / graph: ~1 min including the index build; reranked: 3–6 min first run, seconds once cross-encoder scores are cached |
| Architecture tables | `python -m contract_intelligence.evaluation.compare baseline traditional_rag graph_rag hybrid_rag graph_hybrid_rag` | seconds |
| Improvement loop | `python -m contract_intelligence.improvement.loop` (delete `experiments/improvement_log.jsonl` and `experiments/loop/` to start from scratch) | 2–8 min per iteration |
| Error analysis | `python -m contract_intelligence.evaluation.error_analysis --config configs/graph_hybrid_rag.yaml` | ~10 s |
| Final test metrics | `python -m contract_intelligence.cli --config configs/final.yaml evaluate --split test --reason "reproduce"` | 1–4 min |

Retrieval is deterministic (exact FAISS search, fixed configs, cached reranker scores),
so the train and test numbers in the report reproduce exactly. Latency figures vary
with machine load.

## Assumptions

1. **Metric definitions.** Character-level precision and recall follow the
   LegalBench-RAG authors' code (`QAResult.precision`/`recall` in
   `zeroentropy-ai/legalbenchrag`): overlaps are summed per retrieved snippet, an empty
   retrieval scores 0, and queries are averaged with equal weight. Overlapping chunks
   would count shared characters twice, so merged-span variants are also reported.
   All final architectures use non-overlapping chunks, where the two are equal.
2. **Ranking vs. abstention.** P@k/R@k are computed on the ranked list before
   abstention, which isolates ranking quality and is comparable to the paper.
   Abstention is reported separately in both directions. End-to-end metrics, where an
   abstained query scores 0, are also saved.
3. **Abstention thresholds** are calibrated per configuration on train only (answerable
   train queries vs. train unanswerable queries), maximising balanced accuracy. Each
   pipeline's score has its own scale (cosine, graph score, cross-encoder logit), so
   thresholds are never shared, and no score is treated as a probability.
4. **Unanswerable set.** The initial repository contained 30 copies of one templated
   question. It was replaced with 41 varied, hand-verified queries in three categories
   (clause absent from an indexed contract, agreement not in the corpus, out of scope)
   **before** the improvement loop, in commit `ac3b8b4`, and has not been edited
   since. The Glu Mobile "Amendment No. 3" has no governing-law clause (that clause
   is in the unindexed original agreement), so "what is the governing law?" for it is
   labelled unanswerable.
5. **Corpus.** As the benchmark authors did, the index holds the 29 contracts that the
   194 selected queries refer to, including those referenced only by test queries.
   Their text is indexed, but test queries and labels are never used for tuning.
6. **Queries name their contract.** Benchmark queries start with "Consider the
   <agreement> between <parties>; …". The graph pipelines use this to route to a
   contract. Free-form questions that name no agreement fall back to searching every
   contract.
7. **Text handling.** Files are read with `read_text(encoding="utf-8")` (default
   newline translation), which is what the benchmark offsets assume; all 194 spans
   were verified. Chunk text is never modified, so offsets always point into the
   original file. Whitespace is collapsed only on the copy sent to the models.
8. **Test usage.** Test is evaluated once, for the final frozen configuration. Every
   test run is logged with a reason in `experiments/test_runs.jsonl`.
9. **LLM only for answers, never for retrieval.** Retrieval, graph extraction and
   loop proposals are LLM-free and run locally. Gemini is an optional layer that
   phrases an answer and must quote a retrieved passage verbatim. It was evaluated on
   train only; the frozen retrieval config and its test numbers are unaffected. See
   REPORT.md ("LLM use").
10. **Primary metric.** Recall@5 was chosen for the improvement loop (a reviewer reads
    about five passages), with precision@5 and abstention as guardrails. See REPORT.md.

## Confidentiality

The assignment brief and any text copied from it are not in this repository
(`*.pdf` is git-ignored). No API keys or confidential contracts are stored; the
corpus is the public CUAD dataset.
