from contract_intelligence.config import PipelineConfig, RetrievalConfig
from contract_intelligence.evaluation.runner import K_VALUES, apply_threshold, summarize


def answerable(score: float | None, hit: bool) -> dict:
    gold = [["cuad/a.txt", 100, 200]]
    top = [["cuad/a.txt", 100, 200, 0.9]] if hit else [["cuad/a.txt", 500, 600, 0.9]]
    value = 1.0 if hit else 0.0
    return {
        "query": "q", "answerable": True, "top_score": score, "first_relevant_rank": 1 if hit else None,
        "gold": gold, "retrieved_top": top,
        **{key: {k: value for k in K_VALUES} for key in ("precision", "recall", "precision_merged", "recall_merged")},
        "doc_hit": {k: True for k in K_VALUES},
    }


def unanswerable(score: float, category: str = "out_of_scope") -> dict:
    return {"query": "u", "answerable": False, "category": category, "top_score": score, "retrieved_top": []}


ROWS = [answerable(0.8, True), answerable(0.4, False), answerable(None, False), unanswerable(0.3), unanswerable(0.7)]


def test_threshold_decides_abstention_like_the_retriever() -> None:
    rows = apply_threshold(ROWS, 0.5)
    assert [r["abstained"] for r in rows] == [False, True, True, True, False]
    assert [r["diagnosis"] for r in rows] == [
        "full_coverage", "wrong_abstention", "wrong_abstention", "correct_abstention", "false_answer"
    ]
    assert [r["abstained"] for r in apply_threshold(ROWS, None)] == [False, False, True, False, False]


def test_summary_reports_both_abstention_directions() -> None:
    config = PipelineConfig(name="t", retrieval=RetrievalConfig(top_k=10, abstain_threshold=0.5))
    summary = summarize(config, "train", apply_threshold(ROWS, 0.5))
    a = summary["abstention"]
    assert a["unanswerable_correctly_abstained"] == 0.5
    assert a["unanswerable_wrongly_answered"] == 0.5
    assert a["answerable_wrongly_abstained"] == round(2 / 3, 4)
    assert summary["retrieval"]["recall"][1] == round(1 / 3, 4)  # ranking metrics ignore abstention
    assert summary["end_to_end"]["recall"][1] == round(1 / 3, 4)
