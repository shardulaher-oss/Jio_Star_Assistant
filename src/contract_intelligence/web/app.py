"""Streamlit app. Uses the same Retriever as the CLI and evaluation.

Run with:  streamlit run src/contract_intelligence/web/app.py
"""

from __future__ import annotations

import html
import os
import sys
from dataclasses import replace
from pathlib import Path

# Allow `streamlit run` from a fresh clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from contract_intelligence.answering.generator import AnswerGenerator, load_dotenv  # noqa: E402
from contract_intelligence.cli import with_overrides  # noqa: E402
from contract_intelligence.config import PROJECT_ROOT, AnswerConfig, load_config  # noqa: E402
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


@st.cache_resource(show_spinner=False)
def get_answer_generator(model: str) -> AnswerGenerator:
    return AnswerGenerator(replace(AnswerConfig(), model=model))


def show_in_context(document: str, start: int, end: int, context: int = 400,
                    evidence: tuple[int, int] | None = None) -> None:
    """Passage highlighted in its surrounding text; the answer's quote, if inside, highlighted strongly."""
    def mark(a: int, b: int) -> str:
        if evidence and a <= evidence[0] < evidence[1] <= b:
            return (html.escape(document[a : evidence[0]])
                    + f"<mark style='background:#f5c518;font-weight:600'>{html.escape(document[evidence[0]:evidence[1]])}</mark>"
                    + html.escape(document[evidence[1] : b]))
        return html.escape(document[a:b])

    st.markdown(
        "<div style='white-space: pre-wrap; font-family: monospace; font-size: 0.85em;'>"
        f"<span style='opacity: 0.55'>{html.escape(document[max(0, start - context) : start])}</span>"
        f"<mark>{mark(start, end)}</mark>"
        f"<span style='opacity: 0.55'>{html.escape(document[end : end + context])}</span></div>",
        unsafe_allow_html=True,
    )


def show_answer(answer, corpus_root: str) -> None:
    with st.container(border=True):
        st.subheader("Answer")
        if answer.answer_type == "not_found":
            st.info(f"**{answer.answer}** {answer.note}")
            return
        st.markdown(f"#### {answer.answer}")
        status = "verified quote" if answer.grounded else "quote NOT found in the passages"
        st.caption(f"Type: {answer.answer_type} · confidence: {answer.confidence} · {status} · "
                   f"model: {answer.model}{' (cached)' if answer.cached else ''} · {answer.latency_ms:.0f} ms")
        if answer.evidence:
            e = answer.evidence
            st.markdown(f"**Evidence** (passage #{e.passage_rank}) · `{e.file_path}` · chars `{e.start}–{e.end}`")
            show_in_context(read_document(e.file_path, corpus_root), e.start, e.end, context=300)
        else:
            st.warning(answer.note or "No verifiable evidence was returned.")


def main() -> None:
    st.set_page_config(page_title="Contract Clause Search", layout="wide")
    st.title("Contract Clause Search")
    st.caption(
        "Finds the contract passages that answer a question, with source file and character "
        "offsets, or says so when no passage is relevant enough. Optionally, an LLM reads the top "
        "passages and returns a short answer with a verbatim, offset-checked quote."
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
        st.markdown("**Answer generation**")
        load_dotenv()
        has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        generate = st.checkbox(
            "Generate a precise answer with Gemini", value=has_key, disabled=not has_key,
            help="Sends the top passages (contract text) to Google's Gemini API. Retrieval still "
                 "decides what is shown; the answer must quote a retrieved passage verbatim.",
        )
        model = st.text_input("Gemini model", AnswerConfig().model, disabled=not generate)
        if not has_key:
            st.caption("Set GEMINI_API_KEY in a git-ignored .env file to enable answers.")
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

    answer = None
    if generate:
        try:
            with st.spinner("Asking Gemini for a grounded answer..."):
                answer = get_answer_generator(model).answer(result)
            show_answer(answer, base.data.corpus_root)
        except Exception as error:  # retrieval results stay usable if the API fails
            st.error(f"Answer generation failed: {error}")

    cited = set()
    evidence = None
    if answer is not None and answer.answer_type != "not_found":
        cited = set(answer.supporting_passages) | ({answer.evidence.passage_rank} if answer.evidence else set())
        evidence = (answer.evidence.start, answer.evidence.end) if answer.evidence else None

    st.subheader("Retrieved passages")
    st.caption(result.message + (" Passages cited by the answer are opened in context." if cited else ""))
    for p in result.passages:
        with st.container(border=True):
            tag = " · **cited by the answer**" if p.rank in cited else ""
            st.markdown(f"**#{p.rank}** · score `{p.score:.3f}` · `{p.file_path}` · chars `{p.start}–{p.end}`{tag}")
            st.text(" ".join(p.text.split()))
            with st.expander("Show in context", expanded=p.rank in cited):
                document = read_document(p.file_path, base.data.corpus_root)
                show_in_context(document, p.start, p.end, evidence=evidence)


main()
