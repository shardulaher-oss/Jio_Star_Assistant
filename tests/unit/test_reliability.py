import json
import logging

import pytest

from contract_intelligence.answering.generator import AnswerGenerator
from contract_intelligence.cli import main
from contract_intelligence.config import AnswerConfig
from contract_intelligence.logging_utils import JsonFormatter, query_id
from contract_intelligence.retrieval.abstention import MAX_QUERY_CHARS, clean_query
from contract_intelligence.schemas import Passage, SearchResult


def test_clean_query_strips_control_characters_and_whitespace() -> None:
    assert clean_query("  governing\tlaw\x00 of\n the contract ") == ("governing law of the contract", None)


@pytest.mark.parametrize("query", ["", "   ", "\x00\x01"])
def test_empty_queries_are_rejected(query: str) -> None:
    assert clean_query(query)[1] == "The query is empty."


def test_overlong_queries_are_rejected_not_truncated() -> None:
    text, problem = clean_query("a " * MAX_QUERY_CHARS)
    assert problem.startswith("The query is too long") and len(text) > MAX_QUERY_CHARS


def test_json_logs_carry_fields_but_never_the_query_text() -> None:
    record = logging.makeLogRecord({"name": "contract_intelligence.x", "levelname": "INFO", "msg": "search",
                                    "query_id": query_id("secret clause text"), "query_chars": 18})
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "search" and line["query_chars"] == 18
    assert "secret" not in json.dumps(line) and len(line["query_id"]) == 12


def test_external_llm_is_refused_when_not_allowed(tmp_path) -> None:
    config = AnswerConfig(allow_external=False, cache_path=str(tmp_path / "c.jsonl"))
    generator = AnswerGenerator(config)
    generator.cache_path = tmp_path / "c.jsonl"
    passage = Passage(1, "c", "cuad/a.txt", 0, 10, 1.0, "some text.")
    with pytest.raises(PermissionError, match="allow_external"):
        generator.answer(SearchResult("q", [passage], False, 1.0, 0.0, "ok"))


def test_cli_reports_config_errors_without_a_traceback(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nretrieval: {pipeline: nope}\n", encoding="utf-8")
    assert main(["--config", str(bad), "query", "governing law"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "pipeline must be one of" in err and "Traceback" not in err


def test_cli_rejects_out_of_range_top_k(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["query", "governing law", "--top-k", "0"])
    assert "between 1 and 100" in capsys.readouterr().err
