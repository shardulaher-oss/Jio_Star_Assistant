from dataclasses import replace

import faiss
import networkx as nx
from fakes import HashingEncoder

from contract_intelligence.config import (
    ChunkingConfig,
    GraphConfig,
    HybridConfig,
    PipelineConfig,
    RetrievalConfig,
)
from contract_intelligence.retrieval.graph_hybrid_ranker import GraphHybridRanker
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


def test_graph_hybrid_without_global_dense_stays_in_routed_contract() -> None:
    ranker, dense = make_ranker()
    hybrid = GraphHybridRanker(ranker.encoder, dense, ranker,
                               HybridConfig(bm25_query="question", dense_weight=0.0))
    ranked = hybrid.rank(["Consider the Roanoke Gas franchise with Vinton; when does it expire?"], k=10)[0]
    assert {dense.chunks[p].file_path for p, _ in ranked} == {"cuad/RoanokeGas-Franchise.txt"}
    assert "expire" in dense.chunks[ranked[0][0]].text


def test_graph_hybrid_global_dense_list_can_add_other_contracts() -> None:
    ranker, dense = make_ranker()
    hybrid = GraphHybridRanker(ranker.encoder, dense, ranker, HybridConfig(bm25_query="question"))
    ranked = hybrid.rank(["Consider the Roanoke Gas franchise with Vinton; governed by the laws of"], k=20)[0]
    assert len({dense.chunks[p].file_path for p, _ in ranked}) == 2
    assert dense.chunks[ranked[0][0]].file_path == "cuad/RoanokeGas-Franchise.txt"


def test_abstain_when_query_names_an_unknown_agreement() -> None:
    ranker, dense = make_ranker(replace(SMALL, abstain_when_unrouted=True))
    hybrid = GraphHybridRanker(ranker.encoder, dense, ranker, HybridConfig(bm25_query="question"))
    unknown = "Consider the Distribution Agreement between Vantage Media and Coastal Cinemas; what is the term?"
    known = "Consider the Roanoke Gas franchise with Vinton; when does it expire?"
    assert ranker.rank([unknown], k=5) == [[]] and hybrid.rank([unknown], k=5) == [[]]
    assert ranker.rank([known], k=5)[0] and hybrid.rank([known], k=5)[0]
    assert ranker.rank(["when does the franchise expire?"], k=5)[0]  # names no agreement: search all


def test_retriever_uses_graph_ranker_and_keeps_offsets() -> None:
    ranker, dense = make_ranker()
    config = replace(PipelineConfig(name="g"), retrieval=RetrievalConfig(pipeline="graph", top_k=3))
    result = Retriever(config, ranker.encoder, dense, ranker).search(
        "Consider the Starlight Pictures license; which laws govern"
    )
    for p in result.passages:
        doc = next(d for d in DOCS if d.file_path == p.file_path)
        assert doc.text[p.start : p.end] == p.text


def test_user_chosen_contract_overrides_routing() -> None:
    ranker, dense = make_ranker()
    hybrid = GraphHybridRanker(ranker.encoder, dense, ranker, HybridConfig(bm25_query="question"))
    starlight = {"cuad/StarlightPictures-License.txt"}
    query = "Consider the Roanoke Gas franchise with Vinton; governed by the laws of which state"
    for ranked in (ranker.rank([query], k=10, restrict=starlight)[0], hybrid.rank([query], k=10, restrict=starlight)[0]):
        assert ranked and {dense.chunks[p].file_path for p, _ in ranked} == starlight


def test_retriever_filters_non_graph_pipelines_to_the_chosen_contract() -> None:
    encoder = HashingEncoder()
    _, dense = make_ranker()
    retriever = Retriever(PipelineConfig(name="d"), encoder, dense)  # dense ranker, no restrict support
    result = retriever.search("governed by the laws of", contracts={"cuad/RoanokeGas-Franchise.txt"})
    assert result.passages and {p.file_path for p in result.passages} == {"cuad/RoanokeGas-Franchise.txt"}
