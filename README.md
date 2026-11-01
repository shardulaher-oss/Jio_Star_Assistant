# Contract Intelligence Assistant

This project will implement a measured legal-clause retrieval system over the CUAD subset of LegalBench-RAG. Given a natural-language question, it will return ranked contract passages with their source file and original character offsets, or abstain when no indexed passage is sufficiently relevant. It will expose both a Streamlit web interface and a command-line interface.

The current repository contains the design and implementation blueprint only. The confidential assignment PDF and downloaded benchmark data are intentionally excluded from version control.

## Blueprint

See [`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md) for:

- the problem definition and expected behavior;
- mandatory versus optional scope;
- retrieval/RAG concepts and the proposed architecture;
- module responsibilities and dependency direction;
- risks, assumptions, and the phased delivery roadmap.

## Phase 1 dataset preparation

The preparation command reads the permitted CUAD benchmark, applies the exact
benchmark `mini()` selector, downloads only the referenced contracts, validates
all annotated character spans, assigns the exact SHA-256 train/test split, and
writes reproducible metadata:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m contract_intelligence.data.prepare
```

Expected dataset statistics:

```text
query_count: 194
train_count: 140
test_count: 54
referenced_contract_count: 29
unanswerable_count: 41
unanswerable_train_count: 30
unanswerable_test_count: 11
```

The unanswerable set in `data/queries/unanswerable.json` is hand-written and frozen
before the improvement loop. It has three categories: clauses that are absent from
an indexed contract (verified by keyword and synonym search of that contract),
agreements that are not in the corpus (including near misses that reuse a real
party or title), and out-of-scope questions that no contract can answer.

The source benchmark and contracts are downloaded under ignored `data/raw/`.
The selected query files, query manifest, document manifest, and frozen
unanswerable set are written under tracked `data/processed/` and `data/queries/`.

## Chunking

Pipeline settings live in `configs/*.yaml`. To chunk the corpus and print statistics:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m contract_intelligence.indexing.chunking --config configs/baseline.yaml
```

Chunks are exact slices of the original UTF-8 text (`document[start:end] == chunk.text`),
trimmed of leading and trailing whitespace. No text is rewritten, so offsets stay valid.

Two strategies are available through `chunking.strategy`:

- `fixed`: character windows of `chunk_size` with `chunk_overlap` (the baseline).
- `sentence`: whole sentences and paragraphs packed up to `chunk_size` characters, so
  passages never start or end mid-sentence. `chunk_overlap` is the character budget
  of whole sentences repeated in the next chunk.

Chunking study variants live in `configs/experiments/`. Results are in
`experiments/chunking_study.md`.

## Embeddings and FAISS index

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m contract_intelligence.indexing.dense_index --config configs/baseline.yaml
```

The first run downloads the embedding model (about 90 MB) and takes about a minute on CPU.
Indexes are saved under the git-ignored `artifacts/indexes/<fingerprint>/` (`index.faiss`,
`chunks.jsonl`, `manifest.json`). The fingerprint covers the corpus hashes, the chunking
settings and the embedding model, so a matching index is reused and anything else is
rebuilt. Pass `--rebuild` to force a rebuild.

## Retrieval and abstention

`contract_intelligence.retrieval.retriever.Retriever` is the single backend used by the
CLI, the web app and evaluation. It returns ranked passages (file, start/end offsets,
cosine score, text), or an empty list with a message when the best score is below
`retrieval.abstain_threshold`.

To recalibrate the threshold (train queries and train unanswerable queries only):

```powershell
.\.venv\Scripts\python.exe -m contract_intelligence.retrieval.calibrate --config configs/baseline.yaml
```

## Interfaces

Both interfaces call the same `Retriever`, so they return identical results for the
same config.

**Web app (Streamlit):**

```powershell
.\.venv\Scripts\streamlit.exe run src/contract_intelligence/web/app.py
```

Open http://localhost:8501. The sidebar selects the config, top-k and abstention
threshold. Each result shows the source contract, character offsets and cosine score,
plus an expander that highlights the passage in its surrounding contract text.

**CLI:**

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m contract_intelligence.cli query "Who owns the IP in derivative works?"
.\.venv\Scripts\python.exe -m contract_intelligence.cli query "governing law" --top-k 3 --json
.\.venv\Scripts\python.exe -m contract_intelligence.cli query "termination for convenience" --no-abstain
.\.venv\Scripts\python.exe -m contract_intelligence.cli build-index --config configs/baseline.yaml
.\.venv\Scripts\python.exe -m contract_intelligence.cli evaluate --split train
```

`--config` goes before the subcommand, e.g. `cli --config configs/other.yaml query "..."`.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m contract_intelligence.evaluation.runner --config configs/baseline.yaml --split train
```

This reports character-level precision and recall at k = 1, 2, 4, 8, 16, 32 and 64
(mirroring the LegalBench-RAG authors' implementation), abstention in both directions,
and a per-query failure diagnosis. Results are written to
`experiments/runs/<config>__<split>/`. The train run takes about 15 seconds once the
index exists.

Test runs require `--split test --reason "..."`, and each one is appended to
`experiments/test_runs.jsonl`, so the number of test runs is recorded.

## Planned reproducibility contract

The implementation must select exactly 194 CUAD queries with the benchmark authors' `mini()` procedure, index only contracts referenced by those queries, preserve UTF-8 character offsets, use the specified SHA-256 split, freeze at least 30 unanswerable questions before optimization, and keep all improvement decisions reproducible from configuration and experiment logs.

## Confidentiality

Do not add the assignment PDF, copied assignment text, benchmark secrets, API keys, or confidential contracts to this repository. Any AI-assisted development and all assumptions will be documented in the final README as required by the brief.
