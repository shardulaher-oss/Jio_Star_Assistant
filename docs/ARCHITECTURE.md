# Production architecture and operations

This document is for someone who will run, configure or extend the system. Setup
steps are in the [README](../README.md) and results are in [REPORT.md](../REPORT.md).

## 1. Components

| Concern | Module | Notes |
|---|---|---|
| Data ingestion | `data/prepare.py` | Exact `mini()` selection, SHA-256 split, downloads only the referenced contracts, verifies every gold span, writes manifests |
| Document loading and integrity | `data/corpus.py` | Each contract checked against its manifest SHA-256 and character count before use |
| Preprocessing and segmentation | `indexing/segmentation.py` | Sentence and paragraph units as offsets; the text itself is never modified |
| Chunking | `indexing/chunking.py` | `fixed` or `sentence`; every chunk is an exact slice `document[start:end]` |
| Embedding generation | `indexing/dense_index.py` (`SentenceTransformerEncoder`) | Batched, L2-normalised; LRU cache for query vectors |
| Vector indexing | `indexing/dense_index.py` (`DenseIndex`) | Exact FAISS `IndexFlatIP`, persisted per fingerprint |
| Graph construction | `graph/extraction.py`, `graph/builder.py` | Rule-based, with provenance on every node and edge; NetworkX, gzipped node-link JSON |
| Keyword retrieval | `indexing/bm25_index.py`, `retrieval/lexical.py` | Okapi BM25 over the same chunks |
| Graph retrieval | `retrieval/graph_ranker.py`, `retrieval/query.py` | Contract routing, question-only scoring, edge expansion |
| Hybrid retrieval | `retrieval/fusion.py`, `retrieval/hybrid_ranker.py`, `retrieval/graph_hybrid_ranker.py` | Weighted Reciprocal Rank Fusion |
| Reranking | `retrieval/reranking.py` | Cross-encoder, scores cached on disk |
| Abstention and input validation | `retrieval/abstention.py` | Query cleaning and limits; threshold rule |
| Retrieval service | `retrieval/retriever.py` | The one backend used by the CLI, the app and evaluation |
| Answer generation (optional) | `answering/generator.py` | Gemini structured answer with a verified, offset-mapped quote; cached |
| Evaluation | `evaluation/{metrics,runner,diagnostics,error_analysis,compare,benchmark}.py`, `answering/evaluate.py` | LegalBench-RAG metrics, abstention, cost |
| Improvement loop | `improvement/{diagnose,propose,loop}.py` | Train only; logs every iteration |
| Interfaces | `cli.py`, `web/app.py` | argparse CLI; Streamlit app |
| Cross-cutting | `config.py`, `logging_utils.py`, `schemas.py` | Validated configuration, structured logging, typed records |

The pipeline is chosen by `retrieval.pipeline` (`dense | graph | bm25 | hybrid |
graph_hybrid`) through `retrieval.retriever.build_ranker`. Each ranker returns
`(chunk position, score)` lists, and the `Retriever` turns them into passages with
offsets and applies abstention, so a new architecture is one new ranker class.

### Request flow (final configuration)

```
question ─► abstention.clean_query (reject empty / > 2,000 characters)
         ─► query.parse_query ("Consider the <agreement>;" ⟶ contract part + question)
         ─► GraphRanker.route (party/alias IDF match ⟶ contract set, or all contracts)
         ─► graph ranking (question-only cosine + title boost + expansion) ┐
         ─► BM25 on the question, restricted to the routed contracts       ┴► RRF (k = 60)
         ─► cross-encoder rerank of the top 40 (cached)
         ─► abstention.decide (top-1 logit < −5.6523 ⟶ empty result with a reason)
         ─► [optional] AnswerGenerator: top 5 passages ⟶ Gemini JSON ⟶ quote verified ⟶ offsets
```

## 2. Configuration

All settings live in YAML under `configs/` and are validated by
`config.load_config`. Unknown sections or keys, out-of-range values and malformed
YAML fail with a `ConfigError` that names the file and field. Every committed config
is loaded by the test suite.

| Section | Key settings |
|---|---|
| `data` | `corpus_root`, `document_manifest` |
| `chunking` | `strategy` (`fixed`/`sentence`), `chunk_size`, `chunk_overlap` (< size) |
| `embedding` | `model_name`, `query_prefix`/`passage_prefix`, `normalize_whitespace` (model input only), `batch_size`, `device` |
| `index` / `graph.output_root` | Artifact locations (git-ignored) |
| `retrieval` | `pipeline`, `top_k`, `abstain_threshold` (on that pipeline's score scale; calibrated on train) |
| `graph` | `route_contracts`, `route_min_score`, `route_keep_ratio`, `question_only`, `section_title_weight`, `expand_top`, `expansion_decay`, `abstain_when_unrouted` |
| `hybrid` | `bm25_k1`, `bm25_b`, query mode per stage (`full`/`question`), pool sizes, `rrf_k`, list weights (0 disables a list), `reranker_model` (`""` = none), `rerank_pool`, `rerank_batch_size` |
| `answer` | `provider` (`gemini`), `model`, `max_passages`, `temperature`, `timeout_seconds`, `cache_path`, **`allow_external`** |

Named configurations:
- `baseline.yaml`: the Phase 1 reference.
- `traditional_rag.yaml`, `graph_rag.yaml`, `hybrid_rag.yaml`, `graph_hybrid_rag.yaml`:
  the four architectures.
- **`final.yaml`**: frozen and selected on train; the default for the CLI and the app.
- `configs/experiments/`: study variants.
- `experiments/loop/configs/`: every improvement-loop candidate.

## 3. Performance

| Concern | What the system does |
|---|---|
| Persistent indexes | FAISS index, chunk metadata and manifest under `artifacts/indexes/<fingerprint>/`. The fingerprint covers corpus hashes, chunking and model, so changing retrieval-only settings reuses the index and a stale index is never loaded. Load time is about 0.03 s. |
| Graph storage | Gzipped node-link JSON, 224 KB; builds in 1.5 s and loads in 0.1 s. |
| Embedding cache | Passage vectors are cached by the persistent index. Query vectors use a bounded LRU (4,096) with one batched encode for the uncached queries; results are identical with and without it. |
| Batch work | Passages are embedded in batches of 64. The cross-encoder scores the whole rerank pool in batches (8 on a 16 GB laptop; 32 when memory allows). Evaluation retrieves all queries of a split in one batch. |
| Avoiding recomputation | Cross-encoder scores are cached per (model, query, passage) in `artifacts/cache/`; LLM answers per (model, prompt) in `experiments/llm_cache/` (committed). BM25 rebuilds at startup in under 0.1 s, so it is not persisted. |
| Latency (CPU) | Traditional 7.8 ms, Graph 5.7 ms, Hybrid 888 ms, final 571 ms median per new query; the cross-encoder dominates. Gemini adds about 1.4 s per uncached answer. |
| Memory | About 350 MB for the dense retriever, 540 MB for the final pipeline (two small transformers). |
| Startup | 5–10 s (model loading). The app caches the retriever per config with `st.cache_resource`. |

**Normalisation.** Embeddings are L2-normalised because exact inner-product search on
unit vectors *is* cosine similarity, a property of the retrieval metric. That is
different from scaling tabular features (z-scores, min-max), which this system has
no use for: BM25, cosine and cross-encoder scores are never mixed numerically. RRF
combines ranks precisely to avoid rescaling.

## 4. Reliability

- **Validation.**
  - Configuration: strict keys and ranges (section 2).
  - Queries: control characters removed, whitespace collapsed; empty queries and
    queries over 2,000 characters are rejected with a reason instead of being
    searched or truncated.
  - CLI: `--top-k` limited to 1–100.
- **Data integrity.**
  - Gold spans are verified during preparation.
  - Contracts are checked against SHA-256 hashes and lengths at load time.
  - Every chunk satisfies `document[start:end] == chunk.text` (tested on the real
    corpus).
  - The regression test pins the final rankings.
- **Errors.**
  - Expected problems (bad config, missing data, privacy refusal) give a one-line
    `error:` message from the CLI with exit code 2, and a setup hint in the app.
  - If answer generation fails (network, quota, timeout), the error is logged and
    retrieval results are still shown.
  - Gemini calls time out after 60 s and are retried with backoff on 429, 500, 503
    and 504.
- **Logging.**
  - Library modules log to the `contract_intelligence` logger.
  - `--log-level INFO --log-json` produces one JSON object per event (`search`,
    `answer`) with pipeline, latency, counts, scores and a 12-character query hash.
  - **Query and contract text are never logged.**
- **Reproducibility.**
  - Deterministic exact search, pinned configs and cached reranker and LLM outputs.
  - `evaluation.runner` writes summaries and per-query results; test runs require a
    reason and are appended to `experiments/test_runs.jsonl`.
  - Dependencies are pinned to compatible ranges in `requirements.txt`.

## 5. Security and privacy

- **API keys.** Read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment
  or a **git-ignored** `.env`. Keys are never logged, printed or committed. A
  staged-diff check for key patterns was run before every commit that touched LLM
  code.
- **Contract confidentiality.**
  - Retrieval, reranking and the graph run entirely locally with open models.
  - The **only** path that sends contract text off the machine is the optional
    Gemini answer, which receives the top 5 passages. It is off unless requested
    (CLI `--answer`, or the app toggle, which explains what is sent).
  - `answer.allow_external: false` makes answer generation refuse to call any
    external provider. **Set it for confidential agreements**, or replace the
    provider with a self-hosted model.
  - The committed LLM cache contains passages from the public CUAD corpus only.
- **Logs** contain no query or contract text (section 4).
- **Uploads.** The app does not accept uploaded documents (upload size capped at 1 MB
  in `.streamlit/config.toml`). If uploads are added, they should be:
  - stored outside the repository;
  - hashed and checked for size and type (text and PDF only);
  - extracted in a sandbox;
  - never cached in shared locations;
  - deleted when the session ends.
- **Deployment.**
  - `.streamlit/config.toml` binds the app to `localhost` with XSRF protection and
    telemetry off.
  - There is **no built-in authentication**. For anything beyond a local demo, put
    the app behind an authenticating reverse proxy (SSO), use HTTPS, and restrict
    the corpus per user. Do not just change the bind address.
  - `.env` and `.streamlit/secrets.toml` are git-ignored.

## 6. Technical decisions

| Decision | Reason |
|---|---|
| No FastAPI layer | The CLI and the Streamlit app share the `Retriever` in-process, and no external client needs an HTTP API. A service would add deployment, auth and serialization work without improving the demo or the evaluation. If one is needed later, it is a thin wrapper around `Retriever.search` and `AnswerGenerator.answer`. |
| No microservices, vector database or Neo4j | 1,817 chunks and about 4,000 graph nodes fit in memory; exact FAISS search takes about 1 ms. |
| Rule-based graph and loop proposals | Deterministic, reviewable, with provenance; no contract text leaves the machine. |
| LLM only for answer phrasing, never for retrieval decisions | The answer must quote a retrieved passage verbatim. Using the LLM's "not found" to abstain would raise false abstention from 15% to 32% on train, so passages always stay visible. |
| `gemini-3.1-flash-lite` | `gemini-2.5-flash` is closed to new keys and `gemini-3.8-flash` timed out in testing (November 2026). Flash-lite is stable, fast and sufficient for extraction. The model is configurable. |

## 7. Operations

```powershell
$env:PYTHONPATH = "src"
python -m contract_intelligence.data.prepare                      # data + integrity checks
python -m contract_intelligence.cli build-index                   # or let first use build it
python -m contract_intelligence.cli query "..." --answer --log-level INFO --log-json
.\.venv\Scripts\streamlit.exe run src/contract_intelligence/web/app.py
pytest                       # 171 unit + 7 integration tests (~20 s); pytest -m "not integration" for unit only
```

- **Rebuild artifacts.** Delete `artifacts/` (indexes, graphs, reranker cache). They
  are rebuilt on demand from the corpus and configs.
- **Change a setting.** Copy a config, edit it, and evaluate on train with
  `cli --config <file> evaluate`. The fingerprint logic rebuilds the index only if
  chunking, the model or the corpus changed.
- **Memory.** The cross-encoder needs a few hundred MB per batch. Lower
  `hybrid.rerank_batch_size` if out-of-memory errors appear.

## 8. Known limitations

- Abstention is not validated on test (54.5% correct abstention; 95% CI 27–82% with 11
  negatives). "No answer found" must not be read as "the contract has no such clause".
- Routing assumes the query names its agreement. Free-form questions search every
  contract, and misspelled party names (e.g. "PCQ" vs "PC Quote") defeat routing.
- The LLM answer quotes one passage and can miss qualifications elsewhere (e.g. a
  term that expires in 2006 but continues for some properties until 2008). Gemini
  said "not found" for 45 of 140 answerable train queries.
- No authentication, multi-tenancy or document upload; single-process, CPU-only.
- Graph extraction is heuristic: 319 of 903 cross-references are unresolved.
