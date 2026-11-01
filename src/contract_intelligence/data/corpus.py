"""Load the indexed contracts and verify they match the document manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from contract_intelligence.schemas import Document


def read_document(path: Path) -> str:
    # Default text mode + UTF-8 is what the benchmark offsets assume.
    return path.read_text(encoding="utf-8")


def load_documents(manifest_path: Path, corpus_root: Path) -> list[Document]:
    """Load every contract in the manifest, failing if a file is missing or changed."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = []
    for entry in manifest:
        path = corpus_root / entry["file_path"]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run `python -m contract_intelligence.data.prepare`"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['file_path']} does not match its manifest hash")
        text = read_document(path)
        if len(text) != entry["char_count"]:
            raise ValueError(f"{entry['file_path']} has an unexpected character count")
        documents.append(Document(file_path=entry["file_path"], text=text))
    return documents
