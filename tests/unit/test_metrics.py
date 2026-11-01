import pytest

from contract_intelligence.evaluation import diagnostics as dx
from contract_intelligence.evaluation.metrics import (
    Span,
    first_relevant_rank,
    merge_spans,
    precision,
    precision_merged,
    recall,
    recall_merged,
)

GOLD = [Span("a.txt", 100, 200)]


def test_exact_match_scores_one() -> None:
    assert precision(GOLD, GOLD) == recall(GOLD, GOLD) == 1.0


def test_partial_overlap() -> None:
    retrieved = [Span("a.txt", 150, 350)]  # 50 of 200 retrieved chars are gold
    assert precision(retrieved, GOLD) == pytest.approx(50 / 200)
    assert recall(retrieved, GOLD) == pytest.approx(50 / 100)


def test_same_offsets_in_another_file_do_not_count() -> None:
    retrieved = [Span("b.txt", 100, 200)]
    assert precision(retrieved, GOLD) == recall(retrieved, GOLD) == 0.0


def test_empty_retrieval_scores_zero() -> None:
    assert precision([], GOLD) == recall([], GOLD) == 0.0


def test_precision_averages_over_all_retrieved_characters() -> None:
    retrieved = [Span("a.txt", 100, 200), Span("a.txt", 500, 600)]
    assert precision(retrieved, GOLD) == pytest.approx(0.5)
    assert recall(retrieved, GOLD) == 1.0


def test_multiple_gold_spans() -> None:
    gold = [Span("a.txt", 0, 100), Span("b.txt", 0, 300)]
    retrieved = [Span("a.txt", 0, 100)]
    assert recall(retrieved, gold) == pytest.approx(100 / 400)


def test_official_metric_double_counts_overlapping_chunks_merged_does_not() -> None:
    retrieved = [Span("a.txt", 100, 200), Span("a.txt", 150, 250)]
    assert recall(retrieved, GOLD) == pytest.approx(1.5)  # authors' behaviour
    assert recall_merged(retrieved, GOLD) == 1.0
    assert precision_merged(retrieved, GOLD) == pytest.approx(100 / 150)


def test_merge_spans_keeps_files_apart() -> None:
    spans = [Span("a.txt", 0, 10), Span("b.txt", 5, 15), Span("a.txt", 8, 20)]
    assert merge_spans(spans) == [Span("a.txt", 0, 20), Span("b.txt", 5, 15)]


def test_first_relevant_rank() -> None:
    retrieved = [Span("b.txt", 0, 50), Span("a.txt", 0, 99), Span("a.txt", 150, 160)]
    assert first_relevant_rank(retrieved, GOLD) == 3
    assert first_relevant_rank(retrieved[:2], GOLD) is None


@pytest.mark.parametrize(
    "retrieved,abstained,expected",
    [
        ([Span("a.txt", 100, 200)], True, dx.WRONG_ABSTENTION),
        ([Span("b.txt", 100, 200)], False, dx.WRONG_DOCUMENT),
        ([Span("a.txt", 300, 400)], False, dx.WRONG_SECTION),
        ([Span("a.txt", 150, 400)], False, dx.PARTIAL_COVERAGE),
        ([Span("a.txt", 0, 500)], False, dx.FULL_COVERAGE),
    ],
)
def test_diagnose_answerable(retrieved, abstained, expected) -> None:
    assert dx.diagnose_answerable(retrieved, GOLD, abstained) == expected


def test_diagnose_unanswerable() -> None:
    assert dx.diagnose_unanswerable(True) == dx.CORRECT_ABSTENTION
    assert dx.diagnose_unanswerable(False) == dx.FALSE_ANSWER
