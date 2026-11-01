# Baseline: train metrics and initial error analysis

Config: `configs/baseline.yaml`: fixed 500-character chunks, no overlap,
`all-MiniLM-L6-v2`, exact cosine search, top-k 10, abstention threshold 0.6085.
Split: **train only** (140 answerable benchmark queries, 30 unanswerable queries).
Test has not been run.

Reproduce with:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m contract_intelligence.evaluation.runner --config configs/baseline.yaml --split train
```

Full output: `experiments/runs/baseline__train/` (`summary.json`, `per_query.jsonl`).

## Retrieval, before abstention (n = 140)

Character-level metrics as implemented by the LegalBench-RAG authors. The merged
variants are identical here because the baseline chunks do not overlap.

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Precision | 0.046 | 0.051 | 0.050 | 0.038 | 0.026 | 0.017 | 0.010 |
| Recall | 0.026 | 0.093 | 0.194 | 0.278 | 0.364 | 0.478 | 0.565 |
| Gold contract in top k | 0.993 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

MRR (first chunk overlapping any gold span): 0.198.

## Abstention

| Measure | Value |
|---|---|
| Unanswerable queries correctly abstained (n = 30) | 50.0% |
| Answerable queries wrongly abstained (n = 140) | 7.9% |
| Balanced accuracy | 0.711 |
| Abstained, by unanswerable category | clause absent 2/15, contract not in corpus 8/10, out of scope 5/5 |

## Diagnosis at k = 10

| Outcome | Count |
|---|---|
| Right contract, wrong section | 73 / 140 |
| Partial coverage of the gold span | 39 / 140 |
| Full coverage of the gold span | 17 / 140 |
| Wrong abstention | 11 / 140 |
| Wrong contract | 0 / 140 |
| Unanswerable: correct abstention / false answer | 15 / 15 |

Rank of the first chunk that overlaps a gold span: rank 1 for 9 queries, ranks 2–4
for 34, ranks 5–10 for 18, ranks 11–64 for 40, and not found in the top 64 for 39.

## What is going wrong

**1. The contract title dominates the query embedding (main cause).** Every
benchmark query starts with "Consider the <agreement> between <parties>; …". The
top-1 chunk is the contract's opening chunk (characters 0–500, title and parties)
for 74 of 140 queries. Finding the contract is easy (99% at k = 1), but ranking
the clause inside it is not. Example: *"Glu Mobile Amendment No. 3 … expiration
date?"* The top-1 chunk is the amendment's preamble (score 0.816). The expiry
clause starts at character 598, but the first chunk overlapping it is at rank 14.

**2. Easy clause types fail for the same reason.** Governing-law questions have
the lowest recall@8 (0.11, n = 17), although governing-law clauses use very
consistent wording. Example: the BicycleTX governing-law clause ("…laws of England
and Wales…", characters 53956–54252) first appears at rank 8, below seven chunks
that merely mention the parties.

**3. Abstention cannot tell "right contract" from "right clause".** The top-1
score mostly measures whether the query matches a contract's title. Of 15
absent-clause queries, 13 get a confident false answer. Example: *"PennTex Gas
Transportation Agreement: is there a non-disparagement clause?"* scores 0.642 on
the contract's preamble, above the threshold. All 5 out-of-scope queries are
caught.

**4. Wrong abstentions come from weak title matches.** Example: a BicycleTX
right-of-first-refusal query scores 0.594 and abstains. The gold clause (a share
option grant) does not use the words in the question.

**5. Chunk boundaries cut answers.** The median train gold span is 316
characters, and only 73 of 192 gold spans fit inside a single 500-character
chunk. This caps precision and splits answers across chunks (39 partial-coverage
cases).

## Candidate changes for the improvement loop

1. Separate contract routing from clause search: match the contract from the
   title part, then rank chunks within it using only the question part.
2. Chunk size and overlap sweeps, and clause-aware chunk boundaries.
3. Stronger embedding models with query instructions (e.g. BGE, E5).
4. Hybrid lexical plus dense retrieval, since clause vocabulary such as
   "governed by" and "assign" is very predictable.
5. Abstention on the question-only score instead of the top-1 score of the full
   query.
