import math

import pytest

from contract_intelligence.indexing.bm25_index import BM25Index, tokenize

TEXTS = [
    "This Agreement shall be governed by the laws of the State of New York.",
    "Licensee shall not sublicense or assign this Agreement without consent.",
    "The term of this Agreement is five years and renews automatically.",
    "Licensee may sublicense to affiliates. Sublicense terms must match.",
]


def test_tokenize_lowercases_and_keeps_negations() -> None:
    assert tokenize("Licensee shall NOT sublicense the Content.") == [
        "licensee", "shall", "not", "sublicense", "content"
    ]


def test_matching_document_ranks_first() -> None:
    index = BM25Index(TEXTS)
    assert index.search("which laws govern the agreement governed", k=1)[0][0] == 0
    assert index.search("automatic renewal term years", k=1)[0][0] == 2


def test_term_frequency_raises_score() -> None:
    index = BM25Index(TEXTS)
    ranked = index.search("sublicense", k=4)
    assert [doc for doc, _ in ranked] == [3, 1]


def test_rare_terms_weigh_more_than_common_ones() -> None:
    index = BM25Index(TEXTS)
    assert index.idf["york"] > index.idf["licensee"] > 0


def test_score_matches_formula_for_single_term() -> None:
    index = BM25Index(TEXTS, k1=1.2, b=0.75)
    tf, dl = 1, index.doc_len[0]
    df = 1
    idf = math.log(1 + (4 - df + 0.5) / (df + 0.5))
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / index.avgdl))
    assert index.scores("york")[0] == pytest.approx(expected)


def test_allowed_restricts_candidates_and_unknown_terms_score_nothing() -> None:
    index = BM25Index(TEXTS)
    assert [d for d, _ in index.search("sublicense", k=4, allowed={1})] == [1]
    assert index.search("zebra", k=3) == []
