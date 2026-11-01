import re
from pathlib import Path

import pytest

from contract_intelligence.config import PROJECT_ROOT, ConfigError, load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "configs").rglob("*.yaml")))
def test_every_committed_config_is_valid(path: Path) -> None:
    assert load_config(path).name


def test_final_config_is_the_selected_architecture() -> None:
    config = load_config(PROJECT_ROOT / "configs/final.yaml")
    assert config.retrieval.pipeline == "graph_hybrid"
    assert config.chunking.strategy == "sentence" and config.chunking.chunk_size == 800
    assert config.hybrid.reranker_model and config.retrieval.abstain_threshold is not None


@pytest.mark.parametrize(
    "text,message",
    [
        ("chunking: {chunk_size: 500}\n", "'name'"),
        ("name: x\nretrieval: {topk: 5}\n", "unknown key(s) in 'retrieval': ['topk']"),
        ("name: x\nretreival: {top_k: 5}\n", "unknown section(s) ['retreival']"),
        ("name: x\nretrieval: {pipeline: magic}\n", "pipeline must be one of"),
        ("name: x\nchunking: {chunk_size: 100, chunk_overlap: 100}\n", "chunk_overlap"),
        ("name: x\nhybrid: {rerank_query: everything}\n", "rerank_query must be one of"),
        ("name: x\ngraph: {route_keep_ratio: 0}\n", "route_keep_ratio"),
        ("name: x\nanswer: {provider: someone-else}\n", "provider must be one of"),
        ("name: x\nretrieval: [1, 2]\n", "must be a mapping"),
        ("name: [unclosed\n", "invalid YAML"),
    ],
)
def test_invalid_configs_fail_with_a_clear_message(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(write(tmp_path, text))


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("configs/does_not_exist.yaml")


def test_defaults_fill_missing_sections(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "name: minimal\n"))
    assert config.retrieval.pipeline == "dense" and config.answer.allow_external is True
