from contract_intelligence.evaluation.runner import K_VALUES
from contract_intelligence.improvement.diagnose import diagnose
from contract_intelligence.improvement.loop import apply_changes, bootstrap_prob, decide
from contract_intelligence.improvement.propose import RULES, propose

BASE = {"R@5": 0.50, "P@5": 0.10, "balanced_abstention_acc": 0.70, "false_abstention": 0.10}


def q(query: str, recall5: float, rank, files=("a",), gold="a", abstained=False) -> dict:
    return {
        "query": query, "answerable": True, "abstained": abstained, "first_relevant_rank": rank,
        "recall": {k: recall5 for k in K_VALUES}, "gold": [[gold, 0, 10]],
        "retrieved_top": [[f, 0, 10, 0.5] for f in files],
    }


def test_diagnose_categories() -> None:
    rows = [
        q("a", 1.0, 1), q("b", 0.4, 2), q("c", 0.0, 9), q("d", 0.0, None), q("e", 0.0, None, files=("x",)),
        {"query": "u", "answerable": False, "abstained": False, "category": "contract_not_in_corpus"},
    ]
    d = diagnose(rows)
    assert d["full_coverage"] == d["partial_coverage"] == d["ranked_too_low"] == 1
    assert d["not_retrieved"] == d["wrong_document"] == 1
    assert d["false_answer:contract_not_in_corpus"] == 1


def test_proposer_targets_the_biggest_failure_and_skips_tried_rules() -> None:
    diagnosis = {"false_answer:contract_not_in_corpus": 7, "ranked_too_low": 3}
    assert propose(diagnosis, set()).name == "abstain_when_unrouted"
    assert propose(diagnosis, {"abstain_when_unrouted"}).name == "no_section_title_boost"
    assert propose({}, set()) is None
    assert len({r.name for r in RULES}) == len(RULES)


def test_apply_changes_is_a_copy() -> None:
    raw = {"graph": {"expand_top": 5}}
    assert apply_changes(raw, {"graph.expand_top": 0, "hybrid.rrf_k": 20}) == {
        "graph": {"expand_top": 0}, "hybrid": {"rrf_k": 20}
    }
    assert raw == {"graph": {"expand_top": 5}}


def test_bootstrap_is_paired_and_deterministic() -> None:
    base = [q(str(i), 0.5, 1) for i in range(50)]
    better = [q(str(i), 0.6, 1) for i in range(50)]
    gain, prob = bootstrap_prob(base, better)
    assert round(gain, 6) == 0.1 and prob == 1.0
    assert bootstrap_prob(base, better) == (gain, prob)


def test_accept_on_clear_primary_gain() -> None:
    accepted, reason = decide(BASE, {**BASE, "R@5": 0.53}, 0.03, 0.97)
    assert accepted and reason.startswith("(A)")


def test_reject_small_or_uncertain_gain() -> None:
    assert not decide(BASE, {**BASE, "R@5": 0.505}, 0.005, 0.99)[0]
    assert not decide(BASE, {**BASE, "R@5": 0.53}, 0.03, 0.70)[0]


def test_guardrails_block_acceptance() -> None:
    assert decide(BASE, {**BASE, "R@5": 0.6, "P@5": 0.09}, 0.1, 1.0)[1].startswith("guardrail")
    assert decide(BASE, {**BASE, "R@5": 0.6, "false_abstention": 0.3}, 0.1, 1.0)[1].startswith("guardrail")
    assert decide(BASE, {**BASE, "R@5": 0.6, "balanced_abstention_acc": 0.6}, 0.1, 1.0)[1].startswith("guardrail")


def test_accept_abstention_gain_at_equal_ranking() -> None:
    accepted, reason = decide(BASE, {**BASE, "balanced_abstention_acc": 0.75}, 0.0, 0.0)
    assert accepted and reason.startswith("(B)")
