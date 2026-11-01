from pathlib import Path

import networkx as nx
import pytest

from contract_intelligence.config import load_config
from contract_intelligence.graph.builder import add_document, build_graph, load_graph, save_graph
from contract_intelligence.graph.extraction import (
    extract_definitions,
    extract_parties,
    extract_references,
    extract_sections,
    extract_sentence_entities,
    normalise_section_number,
)
from contract_intelligence.config import ChunkingConfig
from contract_intelligence.indexing.chunking import chunk_document
from contract_intelligence.schemas import Document

ROOT = Path(__file__).resolve().parents[2]
HAS_CORPUS = (ROOT / "data/raw/corpus/cuad").exists()

CONTRACT = (
    "LICENSE AGREEMENT\n\nThis Agreement is made between Starlight Pictures Inc. "
    '("Licensor") and Nimbus Streaming LLC ("Licensee").\n\n'
    '1. Definitions\n\n"Licensed Content" means the films listed in Exhibit A.\n\n'
    "2. Grant\n\nLicensor hereby grants Licensee an exclusive, non-transferable license to stream "
    "the Licensed Content, subject to Section 3.\n\n"
    "3. Restrictions\n\nLicensee shall not sublicense the Licensed Content without the prior "
    "written consent of Licensor.\n\n"
    "4. Term\n\nThis Agreement shall expire on December 31, 2026.\n\n"
    "5. Ownership\n\nAll derivative works shall be the sole and exclusive property of Licensor."
)


def test_section_numbers_are_normalised() -> None:
    assert normalise_section_number("Section 4.2") == "4.2"
    assert normalise_section_number("ARTICLE V") == "5"
    assert normalise_section_number("9.2.") == "9.2"


def test_sections_found_with_titles() -> None:
    sections = {e.key: dict(e.attrs)["title"] for e in extract_sections(CONTRACT)}
    assert sections == {"1": "Definitions", "2": "Grant", "3": "Restrictions", "4": "Term", "5": "Ownership"}


def test_explicit_and_alias_definitions() -> None:
    defs = {e.key: dict(e.attrs)["style"] for e in extract_definitions(CONTRACT)}
    assert defs == {"Licensor": "alias", "Licensee": "alias", "Licensed Content": "explicit"}


def test_references_and_parties() -> None:
    assert [e.key for e in extract_references(CONTRACT)] == ["3"]
    assert {e.key for e in extract_parties(CONTRACT)} == {"Starlight Pictures Inc.", "Nimbus Streaming LLC"}


def test_party_suffix_needs_word_boundary() -> None:
    text = "WALABOT-HOME RESELLER AGREEMENT between Vayyar Imaging Ltd. and others"
    assert {e.key for e in extract_parties(text)} == {"Vayyar Imaging Ltd."}
    assert {e.key for e in extract_parties("Use of Logo. The Company agrees")} == set()


def test_sentence_entities() -> None:
    by_kind = {}
    for e in extract_sentence_entities(CONTRACT):
        by_kind.setdefault(e.kind, []).append(e)
    assert [e.key for e in by_kind["date"]] == ["December 31, 2026"]
    assert dict(by_kind["right"][0].attrs)["flags"] == "exclusive,non-transferable"
    assert by_kind["restriction"][0].key == "sublicense"
    assert by_kind["ip"][0].key == "Licensor"


def test_every_extract_is_a_verbatim_slice() -> None:
    from contract_intelligence.graph.extraction import extract_all

    for e in extract_all(CONTRACT):
        assert CONTRACT[e.start : e.end] == e.text


def test_graph_links_clauses_to_definitions_sections_and_rights() -> None:
    doc = Document("cuad/x.txt", CONTRACT)
    chunks = chunk_document(doc, ChunkingConfig("sentence", 120, 0))
    graph = nx.MultiDiGraph()
    stats = add_document(graph, doc, chunks)
    relations = {(d["relation"], v.split(":")[0]) for _, v, d in graph.edges(data=True)}
    assert {("DEFINES", "definition"), ("DEPENDS_ON", "definition"), ("REFERENCES", "section"),
            ("GRANTS_RIGHT", "right"), ("RESTRICTS", "restriction"), ("EXPIRES_ON", "date"),
            ("OWNED_BY", "definition"), ("PARTY_TO", "contract"), ("BELONGS_TO", "section")} <= relations
    assert stats == {"references_resolved": 1, "references_unresolved": 0}
    for _, _, d in graph.edges(data=True):  # provenance points into the document
        assert 0 <= d["ev_start"] < d["ev_end"] <= len(CONTRACT)


@pytest.mark.skipif(not HAS_CORPUS, reason="corpus not downloaded")
def test_real_graph_round_trips_and_has_valid_provenance(tmp_path: Path) -> None:
    graph, _ = build_graph(load_config(ROOT / "configs/experiments/sentence_800_0.yaml"))
    save_graph(graph, tmp_path / "g.json.gz")
    loaded = load_graph(tmp_path / "g.json.gz")
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    assert loaded.number_of_edges() == graph.number_of_edges()
    lengths = {d["file_path"]: d["end"] for _, d in graph.nodes(data=True) if d["type"] == "contract"}
    for _, d in graph.nodes(data=True):
        assert 0 <= d["start"] <= d["end"] <= lengths[d["file_path"]]
