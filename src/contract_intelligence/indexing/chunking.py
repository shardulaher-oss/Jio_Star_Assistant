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


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split a document into fixed-size character windows with optional overlap."""
    text = document.text
    step = config.chunk_size - config.chunk_overlap
    chunks: list[Chunk] = []
    for window_start in range(0, len(text), step):
        window_end = min(window_start + config.chunk_size, len(text))
        start, end = _trim(text, window_start, window_end)
        if start < end:  # skip whitespace-only windows
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.file_path, start, end),
                    file_path=document.file_path,
                    chunk_index=len(chunks),
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
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

    print(json.dumps({
        "config": config.name,
        "chunking": vars(config.chunking),
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_chars_min_mean_max": [min(lengths), round(sum(lengths) / len(lengths)), max(lengths)],
        "train_gold_spans": len(spans),
        "train_gold_span_chars_median": span_lengths[len(span_lengths) // 2],
        "train_gold_spans_inside_one_chunk": f"{contained}/{len(spans)}",
    }, indent=2))


if __name__ == "__main__":
    main()
