from dataclasses import replace

import faiss
import networkx as nx
from fakes import HashingEncoder

from contract_intelligence.config import ChunkingConfig, GraphConfig, PipelineConfig, RetrievalConfig
from contract_intelligence.graph.builder import add_document
from contract_intelligence.indexing.chunking import chunk_document
from contract_intelligence.indexing.dense_index import DenseIndex
from contract_intelligence.retrieval.graph_ranker import ContractRouter, GraphRanker, tokens
from contract_intelligence.retrieval.query import parse_query
from contract_intelligence.retrieval.retriever import Retriever
from contract_intelligence.schemas import Document

DOCS = [
    Document("cuad/StarlightPictures-License.txt",
             'LICENSE AGREEMENT between Starlight Pictures Inc. ("Licensor") and Nimbus Streaming LLC.\n\n'
             '1. Definitions\n\n"Licensed Content" means the films in Exhibit A.\n\n'
             "2. Governing Law\n\nThis Agreement shall be governed by the laws of California.\n\n"
             "3. Grant\n\nLicensor grants a license to stream the Licensed Content, subject to Section 2."),
    Document("cuad/RoanokeGas-Franchise.txt",
             "GAS FRANCHISE AGREEMENT between the Town of Vinton and Roanoke Gas Company.\n\n"
             "1. Term\n\nThis franchise shall expire twenty years after the effective date.\n\n"
             "2. Governing Law\n\nThis Agreement shall be governed by the laws of Virginia."),
]


# With two contracts a unique name token has IDF log(2) ~ 0.69, so the production
# route_min_score (3.0, tuned for 29 contracts) would never route.
SMALL = GraphConfig(route_min_score=0.5)


def make_ranker(config: GraphConfig = SMALL) -> tuple[GraphRanker, DenseIndex]:
    encoder = HashingEncoder()
    graph = nx.MultiDiGraph()
    chunks = []
    for doc in DOCS:
        doc_chunks = chunk_document(doc, ChunkingConfig("sentence", 90, 0))
        add_document(graph, doc, doc_chunks)
        chunks += doc_chunks
    index = faiss.IndexFlatIP(encoder.dimension)
    index.add(encoder.encode_passages([c.text for c in chunks]))
    dense = DenseIndex(index, chunks, {})
    return GraphRanker(encoder, dense, graph, config), dense


def test_parse_query_splits_benchmark_prefix() -> None:
    parsed = parse_query("Consider the Gas Franchise Agreement between Vinton and Roanoke; What is the term?")
    assert parsed == ("the Gas Franchise Agreement between Vinton and Roanoke", "What is the term?")
    assert parse_query("Who owns derivative works?") == ("", "Who owns derivative works?")


def test_tokens_split_camel_case_and_drop_stopwords() -> None:
    assert tokens("GluMobileInc_Content License Agreement") == {"glu", "mobile", "content", "license"}


def test_router_needs_a_distinctive_match() -> None:
    router = ContractRouter({"a": {"starlight", "pictures", "films"}, "b": {"roanoke", "vinton", "gas"}})
    assert router.route("the Roanoke Gas franchise", min_score=0.5, keep_ratio=0.5) == {"b"}
    assert router.route("who owns derivative works", min_score=0.5, keep_ratio=0.5) is None


def test_routed_query_only_returns_clauses_of_the_named_contract() -> None:
    ranker, dense = make_ranker()
    ranked = ranker.rank(
        ["Consider the Gas Franchise Agreement between Vinton and Roanoke Gas; governed by the laws of which state"], k=10
    )[0]
    files = {dense.chunks[p].file_path for p, _ in ranked}
    assert files == {"cuad/RoanokeGas-Franchise.txt"}
    assert "Virginia" in dense.chunks[ranked[0][0]].text


def test_unrouted_query_searches_every_contract() -> None:
    ranker, dense = make_ranker()
    ranked = ranker.rank(["shall be governed by the laws of"], k=20)[0]
    assert {dense.chunks[p].file_path for p, _ in ranked} == {d.file_path for d in DOCS}


def test_definition_of_a_term_in_the_question_is_pulled_in() -> None:
    ranker, dense = make_ranker(replace(SMALL, section_title_weight=0.0))
    ranked = ranker.rank(["Consider the Starlight Pictures license; can Licensed Content be streamed?"], k=10)[0]
    assert any('"Licensed Content" means' in dense.chunks[p].text for p, _ in ranked)


def test_retriever_uses_graph_ranker_and_keeps_offsets() -> None:
    ranker, dense = make_ranker()
    config = replace(PipelineConfig(name="g"), retrieval=RetrievalConfig(pipeline="graph", top_k=3))
    result = Retriever(config, ranker.encoder, dense, ranker).search(
        "Consider the Starlight Pictures license; which laws govern"
    )
    for p in result.passages:
        doc = next(d for d in DOCS if d.file_path == p.file_path)
        assert doc.text[p.start : p.end] == p.text
