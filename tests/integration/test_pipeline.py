"""End-to-end tests on the real corpus and models (skipped when the data is absent).

Run only these:      pytest -m integration
Skip them (fast):    pytest -m "not integration"
"""

import json
from pathlib import Path

import pytest

from contract_intelligence.cli import main
from contract_intelligence.config import load_config
from contract_intelligence.evaluation.runner import load_split

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "configs/final.yaml"
REFERENCE_RUN = ROOT / "experiments/runs/final__train/per_query.jsonl"
N_REGRESSION = 15

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (ROOT / "data/raw/corpus/cuad").exists(), reason="corpus not downloaded"),
]


@pytest.fixture(scope="module")
def retriever():
    from contract_intelligence.retrieval.retriever import Retriever

    return Retriever.from_config(load_config(FINAL))


@pytest.fixture(scope="module")
def documents():
    from contract_intelligence.data.corpus import load_documents

    config = load_config(FINAL)
    docs = load_documents(config.resolve(config.data.document_manifest), config.resolve(config.data.corpus_root))
    return {d.file_path: d.text for d in docs}


def test_final_config_reproduces_committed_train_results(retriever) -> None:
    """Regression: the frozen config must return exactly the rankings behind REPORT.md."""
    reference = [json.loads(line) for line in REFERENCE_RUN.read_text(encoding="utf-8").splitlines()]
    reference = [r for r in reference if r["answerable"]][:N_REGRESSION]
    ranked = retriever.candidates([r["query"] for r in reference], k=10)
    for ref, passages in zip(reference, ranked):
        got = [[p.file_path, p.start, p.end] for p in passages]
        expected = [r[:3] for r in ref["retrieved_top"]]
        assert got == expected, ref["query"]
        assert passages[0].score == pytest.approx(ref["top_score"], abs=1e-3)


def test_every_returned_passage_is_an_exact_slice_of_its_contract(retriever, documents) -> None:
    config = load_config(FINAL)
    queries = [r["query"] for r in load_split(config, "train")[0][:20]]
    for passages in retriever.candidates(queries, k=10):
        for p in passages:
            assert documents[p.file_path][p.start : p.end] == p.text


def test_abstention_end_to_end(retriever) -> None:
    out_of_scope = retriever.search("Who won the ICC Cricket World Cup in 2011?")
    assert out_of_scope.abstained and out_of_scope.passages == []
    answerable = retriever.search(
        "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; "
        "What is the governing law for this contract?"
    )
    assert not answerable.abstained and answerable.passages
    gold = (63649, 63806)
    assert any(p.start <= gold[0] and gold[1] <= p.end for p in answerable.passages[:3])


def test_rejected_inputs_do_not_search(retriever) -> None:
    results = retriever.search_many(["", "x" * 5000])
    assert [r.abstained for r in results] == [True, True]
    assert "empty" in results[0].message and "too long" in results[1].message


def test_cli_json_output(capsys) -> None:
    assert main(["--config", str(FINAL), "query", "Consider the Endorsement Agreement between ABG-Shaq, LLC "
                 "and Papa John's International, Inc.; What is the governing law for this contract?",
                 "--top-k", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["abstained"] is False and len(payload["passages"]) == 3
    assert {"file_path", "start", "end", "score", "text"} <= set(payload["passages"][0])


def test_cached_llm_answer_is_grounded_without_calling_the_api(retriever) -> None:
    """The committed answer cache reproduces the demo answer offline."""
    from contract_intelligence.answering.generator import AnswerGenerator

    class NoNetwork:
        def generate(self, prompt):  # pragma: no cover - fails the test if reached
            raise AssertionError("the API was called although the answer is cached")

    generator = AnswerGenerator(load_config(FINAL).answer, client=NoNetwork())
    result = retriever.search(
        "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; "
        "What is the governing law for this contract?"
    )
    answer = generator.answer(result)
    assert answer.cached and answer.grounded
    assert (answer.evidence.start, answer.evidence.end) == (63649, 63806)


def test_web_app_renders_results_and_abstention() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src/contract_intelligence/web/app.py"), default_timeout=300).run()
    assert not app.exception
    for toggle in app.toggle:  # tests must never call the external LLM
        if "Gemini" in toggle.label:
            toggle.set_value(False)
    {s.label: s for s in app.selectbox}["Pipeline"].select("configs/traditional_rag.yaml").run()
    app.text_area(key="question").input("Who won the ICC Cricket World Cup in 2011?").run()
    assert not app.exception and app.warning
    app.text_area(key="question").input("This Agreement shall be governed by the laws of").run()
    assert not app.exception and any(s.value == "Retrieved passages" for s in app.subheader)
    assert not any(s.value == "Answer" for s in app.subheader)


def test_web_app_contract_picker_restricts_the_search() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src/contract_intelligence/web/app.py"), default_timeout=300).run()
    for toggle in app.toggle:
        toggle.set_value(False)
    boxes = {s.label: s for s in app.selectbox}
    boxes["Contract"].select(next(o for o in boxes["Contract"].options if "Papa John" in o))
    boxes["Question type"].select("What is the governing law for this contract?").run()
    assert not app.exception
    passages = [m.value for m in app.markdown if m.value.startswith("**#")]
    assert passages and all("PapaJohns" in p for p in passages)
    assert len(app.tabs) == 3 and len(app.dataframe) >= 2  # evaluation tab tables
