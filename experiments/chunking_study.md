# Chunking study: boundaries and overlap (train only)

**Question.** Fixed 500-character windows start and end mid-sentence: only 2 of 192
train gold spans start on a chunk boundary. Does chunking on sentence and paragraph
boundaries, or adding overlap, improve retrieval?

**Setup.** Everything is identical to `configs/baseline.yaml` (all-MiniLM-L6-v2,
exact cosine search, top-k 10) except the `chunking` block. Configs are in
`configs/experiments/`. Train split only: 140 answerable queries (192 gold spans) and
30 unanswerable queries. Test has not been run.

Overlapping chunks make the official LegalBench-RAG metric count shared characters
twice, so variants are compared on the **merged** metrics. These merge the retrieved
spans per file first, and equal the official numbers when chunks do not overlap.

## Boundary alignment

| Chunking | Chunks | Gold spans inside one chunk | Gold starts on a chunk start | Gold ends on a chunk end |
|---|---|---|---|---|
| fixed 500 / 0 (baseline) | 2,439 | 73 | 2 | 3 |
| fixed 500 / 100 | 3,035 | 87 | 1 | 2 |
| sentence 500 / 0 | 3,016 | 134 | 85 | 66 |
| sentence 500 / 150 | 3,698 | 139 | 43 | 72 |
| sentence 800 / 0 | 1,817 | 162 | 52 | 42 |
| sentence 800 / 200 | 2,192 | 170 | 29 | 47 |

The segmenter itself puts 138 of 192 gold starts and 177 of 192 gold ends exactly on a
sentence or paragraph boundary. Most misses are annotation conventions: the gold span
leaves out a leading clause number ("9.2. ") or opening quote, or a heading runs
straight into its text.

## Retrieval (merged metrics, n = 140)

| Config | R@1 | R@4 | R@8 | R@16 | R@64 | P@1 | P@4 | P@8 | MRR | Wrong section @10 | Balanced abstention acc.* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (fixed 500 / 0) | 0.026 | 0.194 | 0.278 | 0.364 | 0.565 | 0.046 | 0.050 | 0.038 | 0.198 | 73 | 0.711 |
| fixed 500 / 100 | 0.029 | 0.177 | 0.286 | 0.397 | 0.622 | 0.040 | 0.051 | 0.040 | 0.202 | 67 | 0.730 |
| sentence 500 / 0 | 0.031 | 0.189 | 0.298 | 0.402 | 0.616 | 0.036 | 0.055 | 0.046 | 0.168 | 72 | 0.723 |
| sentence 500 / 150 | 0.017 | 0.136 | 0.253 | 0.322 | 0.575 | 0.029 | 0.045 | 0.044 | 0.143 | 84 | 0.715 |
| **sentence 800 / 0** | **0.060** | **0.247** | **0.374** | 0.504 | **0.737** | 0.044 | 0.048 | 0.040 | **0.209** | **60** | 0.713 |
| sentence 800 / 200 | 0.048 | 0.234 | 0.372 | **0.533** | 0.735 | 0.034 | 0.045 | 0.038 | 0.195 | 65 | 0.736 |

\* At the threshold calibrated on train for that config (`calibration.json` in each run folder).

## Findings

1. **Sentence boundaries fix the mid-sentence passages.** Passages now begin and end
   on sentence or paragraph boundaries, and many more gold spans fit inside one chunk
   (134 vs 73 at the same 500-character budget).
2. **Larger sentence chunks help most.** Sentence 800 / 0 raises recall@8 from 0.278
   to 0.374 (+0.096) and recall@64 from 0.565 to 0.737. Precision@8 stays about the
   same (0.040 vs 0.038), so the gain is not just from retrieving more characters.
3. **Overlap does not pay off on train.** At 500 characters, 150 characters of overlap
   hurts every metric. Repeated sentences crowd the top-k with near-duplicates of
   the same passage. At 800 characters, 200 of overlap roughly ties no overlap
   (better R@16, worse R@1 and MRR) and costs 21% more chunks. Overlap stays at 0.
4. **The main failure is unchanged.** About 60 queries still retrieve the right
   contract but the wrong section, because the title prefix dominates the query
   embedding. Chunking cannot fix that; it is the target of the hybrid and graph
   work.

**Decision.** Phase 2 architectures use sentence-aware 800-character chunks with no
overlap. `configs/baseline.yaml` is unchanged and remains the reference.
