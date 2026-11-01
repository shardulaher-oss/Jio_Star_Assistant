"""Streamlit app. Uses the same Retriever as the CLI and evaluation.

Run with:  streamlit run src/contract_intelligence/web/app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Allow `streamlit run` from a fresh clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from contract_intelligence.cli import with_overrides  # noqa: E402
from contract_intelligence.config import PROJECT_ROOT, load_config  # noqa: E402
from contract_intelligence.retrieval.retriever import Retriever  # noqa: E402

EXAMPLES = [
    "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; "
    "What is the governing law for this contract?",
    "Does this agreement allow us to sublicense the content to a third-party platform?",
    "Who owns the intellectual property in derivative works?",
    "Who won the ICC Cricket World Cup in 2011?",
]


SCORE_NAMES = {
    "dense": "cosine similarity",
    "graph": "graph score",
    "bm25": "BM25 score",
    "hybrid": "cross-encoder logit or RRF score",
    "graph_hybrid": "cross-encoder logit or RRF score",
}


@st.cache_resource(show_spinner="Loading models and index...")
def get_retriever(config_path: str) -> Retriever:
    return Retriever.from_config(load_config(config_path))


@st.cache_data
def read_document(file_path: str, corpus_root: str) -> str:
    return (PROJECT_ROOT / corpus_root / file_path).read_text(encoding="utf-8")


def show_in_context(document: str, start: int, end: int, context: int = 400) -> None:
    before = html.escape(document[max(0, start - context) : start])
    passage = html.escape(document[start:end])
    after = html.escape(document[end : end + context])
    st.markdown(
        "<div style='white-space: pre-wrap; font-family: monospace; font-size: 0.85em;'>"
        f"<span style='opacity: 0.55'>{before}</span>"
        f"<mark>{passage}</mark>"
        f"<span style='opacity: 0.55'>{after}</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Contract Clause Search", layout="wide")
    st.title("Contract Clause Search")
    st.caption(
        "Finds the contract passages that answer a question, with source file and character "
        "offsets, or says so when no passage is relevant enough. Retrieval only: no generated answers."
    )

    configs = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "configs").glob("*.yaml"))
    with st.sidebar:
        st.header("Settings")
        default = "configs/final.yaml" if "configs/final.yaml" in configs else configs[0]
        config_path = st.selectbox("Pipeline config", configs, index=configs.index(default))
        base = load_config(PROJECT_ROOT / config_path)
        top_k = st.slider("Passages to return (top-k)", 1, 20, base.retrieval.top_k)
        abstain = st.checkbox("Abstain when nothing is relevant enough", value=True)
        default_threshold = base.retrieval.abstain_threshold or 0.0
        # The score depends on the pipeline (cosine, graph score or cross-encoder logit),
        # so the threshold is unbounded.
        threshold = st.number_input(
            f"Abstention threshold ({SCORE_NAMES.get(base.retrieval.pipeline, 'score')})",
            value=float(default_threshold), step=0.01, format="%.4f", disabled=not abstain,
            help="Calibrated on train queries for this config. A relevance score, not a probability.",
        )
        st.markdown("**Examples**")
        for example in EXAMPLES:
            if st.button(example[:70] + ("..." if len(example) > 70 else ""), key=example):
                st.session_state["question"] = example

    question = st.text_area("Question", key="question", height=90,
                            placeholder="e.g. When does the exclusivity period end?")
    if not st.button("Search", type="primary") and not question:
        return

    cached = get_retriever(str(PROJECT_ROOT / config_path))
    # Per-request settings on top of the shared, cached model and index.
    config = with_overrides(base, top_k=top_k, threshold=threshold, no_abstain=not abstain)
    result = Retriever(config, cached.encoder, cached.dense, cached.ranker).search(question)

    if result.abstained:
        st.warning(f"**No relevant clause found.** {result.message}")
        return
    st.success(result.message)
    for p in result.passages:
        with st.container(border=True):
            st.markdown(f"**#{p.rank}** · score `{p.score:.3f}` · `{p.file_path}` · chars `{p.start}–{p.end}`")
            st.text(" ".join(p.text.split()))
            with st.expander("Show in context"):
                document = read_document(p.file_path, base.data.corpus_root)
                show_in_context(document, p.start, p.end)


main()
