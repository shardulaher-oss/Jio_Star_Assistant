import json
from pathlib import Path

import pytest

from contract_intelligence.answering import generator as gen
from contract_intelligence.answering.generator import AnswerGenerator, build_prompt, locate_quote
from contract_intelligence.config import AnswerConfig
from contract_intelligence.schemas import Passage, SearchResult

DOC = (
    "C. Governing Law.\n\nThis Agreement shall be governed by and construed in accordance with the\n"
    "laws of the State of Delaware, without regard to its principles of conflicts of law."
)
P1 = Passage(1, "c1", "cuad/a.txt", 1000, 1000 + 17, 0.9, DOC[:17])
P2 = Passage(2, "c2", "cuad/a.txt", 1019, 1000 + len(DOC), 0.8, DOC[19:])
RESULT = SearchResult("What is the governing law?", [P1, P2], False, 0.9, 0.0, "Returned 2 passages.")


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = 0

    def generate(self, prompt: str) -> dict:
        self.calls += 1
        return self.response


def make(tmp_path: Path, response: dict) -> tuple[AnswerGenerator, FakeClient]:
    client = FakeClient(response)
    config = AnswerConfig(cache_path=str(tmp_path / "cache.jsonl"))
    generator = AnswerGenerator(config, client)
    generator.cache_path = tmp_path / "cache.jsonl"
    return generator, client


GOOD = {
    "answer": "The agreement is governed by Delaware law.", "answer_type": "value",
    "evidence_passage": 2,
    "evidence_quote": "governed by and construed in accordance with the laws of the State of Delaware",
    "supporting_passages": [1, 2], "confidence": "high",
}


def test_quote_is_located_with_exact_document_offsets(tmp_path: Path) -> None:
    answer = make(tmp_path, GOOD)[0].answer(RESULT)
    assert answer.grounded and answer.evidence.passage_rank == 2
    start = answer.evidence.start - 1000
    assert DOC[start : answer.evidence.end - 1000] == answer.evidence.text
    assert answer.evidence.text.startswith("governed by") and answer.evidence.text.endswith("Delaware")


def test_quote_matches_across_line_breaks_and_wrong_passage_number() -> None:
    evidence = locate_quote("laws of the State of Delaware", [P1, P2], preferred=1)
    assert evidence is not None and evidence.passage_rank == 2


def test_paraphrased_quote_is_flagged_ungrounded(tmp_path: Path) -> None:
    answer = make(tmp_path, {**GOOD, "evidence_quote": "Delaware law governs this contract"})[0].answer(RESULT)
    assert not answer.grounded and answer.evidence is None and "unverified" in answer.note


def test_not_found_answer(tmp_path: Path) -> None:
    response = {**GOOD, "answer_type": "not_found", "evidence_passage": 0, "evidence_quote": ""}
    answer = make(tmp_path, response)[0].answer(RESULT)
    assert answer.answer_type == "not_found" and answer.evidence is None


def test_abstained_retrieval_never_calls_the_model(tmp_path: Path) -> None:
    generator, client = make(tmp_path, GOOD)
    answer = generator.answer(SearchResult("q", [], True, 0.1, 0.5, "below threshold"))
    assert answer.answer_type == "not_found" and client.calls == 0


def test_responses_are_cached_on_disk(tmp_path: Path) -> None:
    generator, client = make(tmp_path, GOOD)
    first = generator.answer(RESULT)
    second = generator.answer(RESULT)
    assert client.calls == 1 and not first.cached and second.cached
    reloaded, client2 = make(tmp_path, GOOD)
    assert reloaded.answer(RESULT).cached and client2.calls == 0
    record = json.loads((tmp_path / "cache.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["model"] == "gemini-3.1-flash-lite"


def test_prompt_numbers_passages_with_sources() -> None:
    prompt = build_prompt("Q?", [P1, P2])
    assert "[1] source: cuad/a.txt (characters 1000-1017)" in prompt and "[2]" in prompt


def test_dotenv_does_not_override_existing_environment(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('A_TEST_KEY="from-file"\nB_TEST_KEY=x\n', encoding="utf-8")
    monkeypatch.setenv("A_TEST_KEY", "from-env")
    monkeypatch.delenv("B_TEST_KEY", raising=False)
    gen.load_dotenv(env)
    import os

    assert os.environ["A_TEST_KEY"] == "from-env" and os.environ["B_TEST_KEY"] == "x"
    monkeypatch.delenv("B_TEST_KEY")


def test_invalid_model_output_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        make(tmp_path, {**GOOD, "answer_type": "maybe"})[0].answer(RESULT)
