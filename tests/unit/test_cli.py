import pytest

from contract_intelligence.cli import build_parser, format_result, with_overrides
from contract_intelligence.config import PipelineConfig, RetrievalConfig
from contract_intelligence.schemas import Passage, SearchResult

CONFIG = PipelineConfig(name="t", retrieval=RetrievalConfig(top_k=10, abstain_threshold=0.6))


def test_query_arguments_parse() -> None:
    args = build_parser().parse_args(["query", "governing law?", "--top-k", "3", "--json"])
    assert (args.question, args.top_k, args.json, args.command) == ("governing law?", 3, True, "query")


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_overrides_only_change_what_is_given() -> None:
    assert with_overrides(CONFIG).retrieval == CONFIG.retrieval
    assert with_overrides(CONFIG, top_k=3).retrieval == RetrievalConfig(3, 0.6)
    assert with_overrides(CONFIG, threshold=0.7).retrieval.abstain_threshold == 0.7
    assert with_overrides(CONFIG, no_abstain=True).retrieval.abstain_threshold is None


def test_format_shows_source_offsets_and_score() -> None:
    passage = Passage(1, "id", "cuad/a.txt", 120, 180, 0.73, "This Agreement shall be governed\n by Delaware law.")
    text = format_result(SearchResult("q", [passage], False, 0.73, 0.6, "Returned 1 passages."))
    assert "#1  score 0.730  cuad/a.txt  [chars 120-180]" in text
    assert "governed by Delaware law." in text


def test_format_explains_abstention() -> None:
    text = format_result(SearchResult("q", [], True, 0.41, 0.6, "best score 0.410 is below the threshold"))
    assert "No answer found." in text and "below the threshold" in text
