"""Prepare the fixed LegalBench-RAG CUAD-mini dataset."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

DATASET_BASE_URL = (
    "https://huggingface.co/datasets/awinml/legalbench-rag/resolve/main/"
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    answer: str


@dataclass(frozen=True)
class QueryRecord:
    query: str
    snippets: tuple[dict[str, object], ...]
    split: str
    answerable: bool = True


def mini(tests: list[dict[str, object]], n: int = 194) -> list[dict[str, object]]:
    """Select the benchmark author's deterministic CUAD-mini subset."""
    if len(tests) <= n:
        return tests

    def key(test: dict[str, object]) -> float:
        snippets = test["snippets"]
        first = snippets[0]
        random.seed(first["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    digest = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if digest % 10 < 3 else "train"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_snippets(
    selected: list[dict[str, object]], corpus_root: Path
) -> None:
    errors: list[str] = []
    for query_record in selected:
        for snippet in query_record["snippets"]:
            relative_path = Path(str(snippet["file_path"]))
            document_path = corpus_root / relative_path
            if not document_path.exists():
                errors.append(f"{query_record['query']}: missing {relative_path}")
                continue
            document = document_path.read_text(encoding="utf-8")
            start, end = snippet["span"]
            answer = str(snippet["answer"])
            if document[int(start) : int(end)] != answer:
                errors.append(
                    f"{query_record['query']}: {relative_path} "
                    f"[{start}:{end}] does not match the answer"
                )
    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"{len(errors)} span validation errors:\n{preview}")


def _download_contracts(
    selected: list[dict[str, object]], corpus_root: Path
) -> list[str]:
    paths = sorted(
        {
            str(snippet["file_path"])
            for record in selected
            for snippet in record["snippets"]
        }
    )
    for relative_path in paths:
        destination = corpus_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        url = DATASET_BASE_URL + quote("corpus/" + relative_path, safe="/")
        urlretrieve(url, destination)
    return paths


def prepare(
    benchmark_path: Path,
    corpus_root: Path,
    manifest_path: Path,
    negatives_path: Path,
    download: bool = False,
) -> dict[str, object]:
    benchmark = _read_json(benchmark_path)
    selected = mini(list(benchmark["tests"]))
    if len(selected) != 194:
        raise ValueError(f"Expected 194 selected queries, got {len(selected)}")

    referenced_paths = (
        _download_contracts(selected, corpus_root) if download else sorted(
            {
                str(snippet["file_path"])
                for record in selected
                for snippet in record["snippets"]
            }
        )
    )
    _validate_snippets(selected, corpus_root)

    records = [
        QueryRecord(
            query=str(record["query"]),
            snippets=tuple(record["snippets"]),
            split=assign_split(str(record["query"])),
        )
        for record in selected
    ]
    manifest = {
        "selection": "LegalBench-RAG CUAD mini",
        "query_count": len(records),
        "train_count": sum(record.split == "train" for record in records),
        "test_count": sum(record.split == "test" for record in records),
        "referenced_contract_count": len(referenced_paths),
        "referenced_contracts": referenced_paths,
        "queries": [asdict(record) for record in records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    query_dir = manifest_path.parent / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    for name, query_records in (
        ("selected.json", records),
        ("train.json", [record for record in records if record.split == "train"]),
        ("test.json", [record for record in records if record.split == "test"]),
    ):
        (query_dir / name).write_text(
            json.dumps([asdict(record) for record in query_records], indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    negatives_path.parent.mkdir(parents=True, exist_ok=True)
    if not negatives_path.exists():
        negatives = [
            {
                "id": f"unanswerable-{index:03d}",
                "query": query,
                "answerable": False,
                "split": assign_split(query),
                "reason": "The named agreement is not among the indexed CUAD contracts.",
            }
            for index, query in enumerate(
                [
                    f"What is the governing law of the fictional Aurora Moon Streaming Agreement {i}?"
                    for i in range(1, 31)
                ],
                1,
            )
        ]
        negatives_path.write_text(
            json.dumps(negatives, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return {
        "query_count": len(records),
        "train_count": manifest["train_count"],
        "test_count": manifest["test_count"],
        "referenced_contract_count": len(referenced_paths),
        "unanswerable_count": 30,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    result = prepare(
        benchmark_path=root / "data/raw/benchmarks/cuad.json",
        corpus_root=root / "data/raw/corpus",
        manifest_path=root / "data/processed/cuad_mini_manifest.json",
        negatives_path=root / "data/queries/unanswerable.json",
        download=True,
    )
    print(json.dumps(result, indent=2))
