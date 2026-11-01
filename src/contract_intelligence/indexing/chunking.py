"""Offset-preserving chunking.

Chunk text is never rewritten: every chunk is a slice of the original document, so
its (start, end) can be scored directly against the benchmark's character spans.
Any normalisation for embedding happens later, on a copy of the text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable

from contract_intelligence.config import ChunkingConfig, load_config
from contract_intelligence.data.corpus import load_documents
from contract_intelligence.indexing.segmentation import segment, split_long_unit
from contract_intelligence.schemas import Chunk, Document


def make_chunk_id(file_path: str, start: int, end: int) -> str:
    doc_key = hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:10]
    return f"{doc_key}:{start}-{end}"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) to exclude leading and trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _make_chunk(document: Document, index: int, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(document.file_path, start, end),
        file_path=document.file_path,
        chunk_index=index,
        start=start,
        end=end,
        text=document.text[start:end],
    )


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    if config.strategy == "sentence":
        return chunk_sentences(document, config)
    return chunk_fixed(document, config)


def chunk_sentences(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Pack whole sentence/paragraph units into chunks of at most chunk_size characters.

    Chunks start and end on unit boundaries, so passages do not begin or end
    mid-sentence. Units longer than chunk_size are split at whitespace. Overlap
    repeats the trailing whole units of a chunk (up to chunk_overlap characters)
    at the start of the next one.
    """
    text = document.text
    units = [
        piece
        for start, end in segment(text)
        for piece in split_long_unit(text, start, end, config.chunk_size)
    ]
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and units[j + 1][1] - units[i][0] <= config.chunk_size:
            j += 1
        chunks.append(_make_chunk(document, len(chunks), units[i][0], units[j][1]))
        if j == len(units) - 1:
            break
        next_i = j + 1
        while next_i - 1 > i and units[j][1] - units[next_i - 1][0] <= config.chunk_overlap:
            next_i -= 1
        i = next_i
    return chunks


def chunk_fixed(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split a document into fixed-size character windows with optional overlap."""
    text = document.text
    step = config.chunk_size - config.chunk_overlap
    chunks: list[Chunk] = []
    for window_start in range(0, len(text), step):
        window_end = min(window_start + config.chunk_size, len(text))
        start, end = _trim(text, window_start, window_end)
        if start < end:  # skip whitespace-only windows
            chunks.append(_make_chunk(document, len(chunks), start, end))
        if window_end == len(text):
            break
    return chunks


def chunk_corpus(documents: Iterable[Document], config: ChunkingConfig) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, config)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the corpus and print statistics.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    documents = load_documents(
        config.resolve(config.data.document_manifest), config.resolve(config.data.corpus_root)
    )
    chunks = chunk_corpus(documents, config.chunking)
    lengths = [chunk.end - chunk.start for chunk in chunks]

    # How often a gold span fits entirely inside one chunk: an upper bound on how
    # cleanly a single retrieved chunk can cover an answer at this chunk size.
    selected = json.loads(
        config.resolve("data/processed/queries/train.json").read_text(encoding="utf-8")
    )
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.file_path, []).append(chunk)
    spans = [s for record in selected for s in record["snippets"]]
    contained = sum(
        any(c.start <= s["span"][0] and s["span"][1] <= c.end for c in by_doc[s["file_path"]])
        for s in spans
    )
    span_lengths = sorted(s["span"][1] - s["span"][0] for s in spans)
    # How often a gold span starts / ends exactly on a chunk boundary, i.e. chunks
    # cut where the annotators considered a clause to begin or end.
    starts = {(c.file_path, c.start) for c in chunks}
    ends = {(c.file_path, c.end) for c in chunks}
    texts = {d.file_path: d.text for d in documents}
    trimmed = [
        _trim(texts[s["file_path"]], s["span"][0], s["span"][1]) + (s["file_path"],) for s in spans
    ]
    start_aligned = sum((f, a) in starts for a, _, f in trimmed)
    end_aligned = sum((f, b) in ends for _, b, f in trimmed)

    print(json.dumps({
        "config": config.name,
        "chunking": vars(config.chunking),
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_chars_min_mean_max": [min(lengths), round(sum(lengths) / len(lengths)), max(lengths)],
        "train_gold_spans": len(spans),
        "train_gold_span_chars_median": span_lengths[len(span_lengths) // 2],
        "train_gold_spans_inside_one_chunk": f"{contained}/{len(spans)}",
        "train_gold_starts_on_chunk_start": f"{start_aligned}/{len(spans)}",
        "train_gold_ends_on_chunk_end": f"{end_aligned}/{len(spans)}",
    }, indent=2))


if __name__ == "__main__":
    main()
