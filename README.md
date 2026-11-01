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

## Planned reproducibility contract

The implementation must select exactly 194 CUAD queries with the benchmark authors' `mini()` procedure, index only contracts referenced by those queries, preserve UTF-8 character offsets, use the specified SHA-256 split, freeze at least 30 unanswerable questions before optimization, and keep all improvement decisions reproducible from configuration and experiment logs.

## Confidentiality

Do not add the assignment PDF, copied assignment text, benchmark secrets, API keys, or confidential contracts to this repository. Any AI-assisted development and all assumptions will be documented in the final README as required by the brief.
