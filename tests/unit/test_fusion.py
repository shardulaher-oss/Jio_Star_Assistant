import pytest

from contract_intelligence.retrieval.fusion import reciprocal_rank_fusion


def test_items_found_by_both_rankers_rise() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]], k=60)
    assert [item for item, _ in fused][:2] == [1, 3]
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 63)


def test_single_ranking_preserves_order() -> None:
    assert [i for i, _ in reciprocal_rank_fusion([[5, 2, 9]])] == [5, 2, 9]


def test_weights_shift_preference() -> None:
    fused = reciprocal_rank_fusion([[1, 2], [2, 1]], k=1, weights=[3.0, 1.0])
    assert fused[0][0] == 1


def test_ties_break_deterministically_and_empty_inputs_work() -> None:
    assert reciprocal_rank_fusion([[7], [3]]) == [(3, 1 / 61), (7, 1 / 61)]
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_must_match_rankings() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])
