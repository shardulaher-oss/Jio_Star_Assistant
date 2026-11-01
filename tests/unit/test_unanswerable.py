import json
from pathlib import Path

from contract_intelligence.data.prepare import assign_split, load_unanswerable

ROOT = Path(__file__).resolve().parents[2]
NEGATIVES = ROOT / "data/queries/unanswerable.json"
MANIFEST = ROOT / "data/processed/document_manifest.json"


def test_unanswerable_set_is_valid() -> None:
    negatives = load_unanswerable(NEGATIVES)
    assert len(negatives) >= 30
    assert all(n["split"] == assign_split(n["query"]) for n in negatives)


def test_unanswerable_set_covers_several_failure_types() -> None:
    categories = {n["category"] for n in load_unanswerable(NEGATIVES)}
    assert categories == {
        "clause_absent_from_indexed_contract",
        "contract_not_in_corpus",
        "out_of_scope",
    }


def test_absent_clause_queries_target_indexed_contracts() -> None:
    indexed = {d["file_path"] for d in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    for negative in load_unanswerable(NEGATIVES):
        if negative["category"] == "clause_absent_from_indexed_contract":
            assert negative["target_contract"] in indexed
