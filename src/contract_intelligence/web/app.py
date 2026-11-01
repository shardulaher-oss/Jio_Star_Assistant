"""Streamlit app. Uses the same Retriever as the CLI and evaluation.

Run with:  streamlit run src/contract_intelligence/web/app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

# Allow `streamlit run` from a fresh clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from contract_intelligence.answering.generator import AnswerGenerator, load_dotenv  # noqa: E402
from contract_intelligence.cli import with_overrides  # noqa: E402
from contract_intelligence.config import PROJECT_ROOT, AnswerConfig, ConfigError, load_config  # noqa: E402
from contract_intelligence.retrieval.query import parse_query  # noqa: E402
from contract_intelligence.retrieval.retriever import Retriever  # noqa: E402

ALL_CONTRACTS = "All contracts (route by the question)"
OWN_QUESTION = "Write my own question"
EXAMPLES = [
    "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; "
    "What is the governing law for this contract?",
    "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile "
    "Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?",
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
ARCHITECTURES = {
    "baseline": "Phase 1 baseline",
    "traditional_rag": "Traditional RAG",
    "graph_rag": "Graph RAG",
    "hybrid_rag": "Hybrid RAG",
    "graph_hybrid_rag": "Graph-Hybrid RAG",
}


# --- cached resources -----------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and index...")
def get_retriever(config_path: str) -> Retriever:
    return Retriever.from_config(load_config(config_path))


@st.cache_resource(show_spinner=False)
def get_answer_generator(model: str) -> AnswerGenerator:
    return AnswerGenerator(replace(AnswerConfig(), model=model))


@st.cache_data
def read_document(file_path: str, corpus_root: str) -> str:
    return (PROJECT_ROOT / corpus_root / file_path).read_text(encoding="utf-8")


@st.cache_data
def contract_files(manifest: str) -> list[str]:
    return [d["file_path"] for d in json.loads((PROJECT_ROOT / manifest).read_text(encoding="utf-8"))]


@st.cache_data
def question_templates() -> list[str]:
    """Question types, most common first, taken from the TRAIN queries' wording."""
    path = PROJECT_ROOT / "data/processed/queries/train.json"
    if not path.exists():
        return []
    queries = json.loads(path.read_text(encoding="utf-8"))
    return [q for q, _ in Counter(parse_query(r["query"]).question for r in queries).most_common()]


@st.cache_data
def load_summary(name: str, split: str) -> dict | None:
    path = PROJECT_ROOT / f"experiments/runs/{name}__{split}/summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def contract_titles() -> dict[str, str]:
    """file path -> plain-English contract name, from how the TRAIN queries describe it."""
    path = PROJECT_ROOT / "data/processed/queries/train.json"
    titles: dict[str, str] = {}
    if path.exists():
        for record in json.loads(path.read_text(encoding="utf-8")):
            name = re.sub(r"^the\s+", "", parse_query(record["query"]).contract, flags=re.I)
            if name:
                titles.setdefault(record["snippets"][0]["file_path"], name)
    return titles


def contract_label(file_path: str) -> str:
    title = contract_titles().get(file_path)
    if title:
        return title if len(title) <= 110 else title[:107] + "..."
    return filename_label(file_path)


def filename_label(file_path: str) -> str:
    """Fallback name from the filing name: company and agreement type."""
    stem = Path(file_path).stem
    parts = re.split(r"_\d|-\d|-EX-|_EX-", stem)
    company = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", parts[0]).replace("_", " ").strip(" -,")
    agreement = re.split(r"-EX-[\d.]+-|_EX-[\d.]+_", stem)[-1].replace("_", " ").strip(" -")
    agreement = re.sub(r"\d+$", "", agreement).strip()
    if company.lower().replace(" ", "") in agreement.lower().replace(" ", ""):
        return agreement.title() if agreement.isupper() else agreement
    return f"{company.title() if company.isupper() else company} · {agreement.title() if agreement.isupper() else agreement}"


# --- rendering helpers ----------------------------------------------------------

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
        "<div style='white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.82em; "
        "line-height: 1.45; max-height: 360px; overflow-y: auto; padding: 0.5rem; "
        "border: 1px solid rgba(128,128,128,0.25); border-radius: 6px;'>"
        f"<span style='opacity: 0.55'>{html.escape(document[max(0, start - context) : start])}</span>"
        f"<mark style='background: rgba(31,119,180,0.18)'>{mark(start, end)}</mark>"
        f"<span style='opacity: 0.55'>{html.escape(document[end : end + context])}</span></div>",
        unsafe_allow_html=True,
    )


def show_answer(answer, corpus_root: str) -> None:
    with st.container(border=True):
        st.subheader("Answer")
        if answer.answer_type == "not_found":
            st.info(f"**{answer.answer}** {answer.note} The retrieved passages are listed below for review.")
            return
        st.markdown(f"#### {answer.answer}")
        badge = ":green[verified quote]" if answer.grounded else ":red[quote not found in the passages]"
        st.markdown(f"{badge} · type **{answer.answer_type}** · confidence **{answer.confidence}** · "
                    f"{answer.model}{' (cached)' if answer.cached else ''} · {answer.latency_ms:.0f} ms")
        if answer.evidence:
            e = answer.evidence
            st.markdown(f"**Evidence** from passage #{e.passage_rank} · {contract_label(e.file_path)} · "
                        f"characters `{e.start}–{e.end}`")
            show_in_context(read_document(e.file_path, corpus_root), e.start, e.end, context=300,
                            evidence=(e.start, e.end))
        else:
            st.warning(answer.note or "No verifiable evidence was returned.")


def relevance_bars(scores: list[float]) -> list[float]:
    """Display-only bar lengths relative to the best and worst returned score."""
    hi, lo = max(scores), min(scores)
    return [1.0 if hi == lo else 0.15 + 0.85 * (s - lo) / (hi - lo) for s in scores]


# --- pages ----------------------------------------------------------------------

def search_page(config_path: str, base, top_k: int, abstain: bool, threshold: float,
                generate: bool, model: str) -> None:
    files = contract_files(base.data.document_manifest)
    labels = {contract_label(f): f for f in files}
    left, right = st.columns([3, 2])
    with left:
        chosen = st.selectbox("Contract", [ALL_CONTRACTS] + sorted(labels),
                              help="Choosing a contract searches only that agreement, which is the "
                                   "most reliable way to ask about a specific contract.")
    with right:
        template = st.selectbox("Question type", [OWN_QUESTION] + question_templates(),
                                help="Common clause questions from the CUAD benchmark.")
    if template != OWN_QUESTION and st.session_state.get("_template") != template:
        st.session_state["question"] = template
        st.session_state["_template"] = template

    question = st.text_area("Question", key="question", height=80,
                            placeholder="e.g. Does this agreement allow sublicensing to a third party?")
    st.button("Search", type="primary")
    if not question.strip():
        st.caption("Pick a contract and a question type, type a question, or try an example in the sidebar.")
        return

    try:
        cached = get_retriever(str(PROJECT_ROOT / config_path))
    except (ConfigError, FileNotFoundError, ValueError) as error:
        st.error(f"Could not load `{config_path}`: {error}")
        st.info("Run `python -m contract_intelligence.data.prepare` to download and verify the data, "
                "then retry. See the README for setup.")
        st.stop()

    contracts = {labels[chosen]} if chosen != ALL_CONTRACTS else None
    config = with_overrides(base, top_k=top_k, threshold=threshold, no_abstain=not abstain)
    started = time.perf_counter()
    result = Retriever(config, cached.encoder, cached.dense, cached.ranker).search(question, contracts=contracts)
    latency = (time.perf_counter() - started) * 1000

    cols = st.columns(4)
    cols[0].metric("Passages", len(result.passages))
    cols[1].metric("Top score", f"{result.top_score:.3f}" if result.top_score is not None else "–")
    cols[2].metric("Threshold", f"{result.threshold:.3f}" if result.threshold is not None else "off")
    cols[3].metric("Retrieval time", f"{latency:.0f} ms")
    if contracts:
        st.caption(f"Searching only in: **{chosen}**")

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
            st.error(f"Answer generation failed: {error}. Retrieval results are shown below.")

    cited, evidence = set(), None
    if answer is not None and answer.answer_type != "not_found":
        cited = set(answer.supporting_passages) | ({answer.evidence.passage_rank} if answer.evidence else set())
        evidence = (answer.evidence.start, answer.evidence.end) if answer.evidence else None

    header, download = st.columns([4, 1])
    header.subheader("Retrieved passages")
    payload = asdict(result) | ({"answer": answer.to_dict()} if answer else {})
    download.download_button("Download JSON", json.dumps(payload, indent=2, ensure_ascii=False),
                             file_name="contract_search_result.json", mime="application/json",
                             on_click="ignore")
    st.caption(result.message + (" Passages cited by the answer are opened in context." if cited else ""))

    bars = relevance_bars([p.score for p in result.passages])
    for p, bar in zip(result.passages, bars):
        with st.container(border=True):
            title, score = st.columns([5, 1])
            tag = " · :orange[cited by the answer]" if p.rank in cited else ""
            title.markdown(f"**#{p.rank} · {contract_label(p.file_path)}**{tag}  \n"
                           f"`{p.file_path}` · characters `{p.start}–{p.end}`")
            score.metric("score", f"{p.score:.2f}", label_visibility="collapsed")
            st.progress(bar)
            st.markdown(f"<div style='font-size:0.92em'>{html.escape(' '.join(p.text.split()))}</div>",
                        unsafe_allow_html=True)
            with st.expander("Show in context", expanded=p.rank in cited):
                show_in_context(read_document(p.file_path, base.data.corpus_root), p.start, p.end, evidence=evidence)


def evaluation_page() -> None:
    st.markdown("Results from the committed evaluation runs (`experiments/runs/`). Character-level "
                "precision and recall as defined by LegalBench-RAG; test was evaluated once per config.")
    rows = []
    for name, split in (("baseline", "train"), ("final", "train"), ("baseline", "test"), ("final", "test")):
        s = load_summary(name, split)
        if s:
            r, a = s["retrieval"], s["abstention"]
            rows.append({"config": name, "split": split, "n": s["n_answerable"],
                         **{f"P@{k}": r["precision"][str(k)] for k in (1, 3, 5, 10)},
                         **{f"R@{k}": r["recall"][str(k)] for k in (1, 3, 5, 10)},
                         "MRR": r["mrr"], "correct abstention": a["unanswerable_correctly_abstained"],
                         "false abstention": a["answerable_wrongly_abstained"]})
    if rows:
        st.subheader("Baseline vs final")
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True, width="stretch")

    arch = {label: load_summary(name, "train") for name, label in ARCHITECTURES.items()}
    arch = {label: s for label, s in arch.items() if s}
    if arch:
        st.subheader("Architectures (train, n = 140)")
        chart = pd.DataFrame({label: {f"R@{k}": s["retrieval"]["recall"][str(k)] for k in (1, 3, 5, 10)}
                              for label, s in arch.items()})
        st.bar_chart(chart.T, stack=False, horizontal=True, height=320)
        table = pd.DataFrame([{"architecture": label, "P@5": s["retrieval"]["precision"]["5"],
                               "R@5": s["retrieval"]["recall"]["5"], "MRR": s["retrieval"]["mrr"],
                               "balanced abstention": s["abstention"]["balanced_accuracy"]}
                              for label, s in arch.items()])
        st.dataframe(table.style.format(precision=3), hide_index=True, width="stretch")

    log_path = PROJECT_ROOT / "experiments/improvement_log.jsonl"
    if log_path.exists():
        st.subheader("Improvement loop (train only)")
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.dataframe(pd.DataFrame([{
            "#": e["iteration"], "change": e.get("rule", e["kind"]), "R@5": e["metrics"]["R@5"],
            "P@5": e["metrics"]["P@5"], "balanced abstention": e["metrics"]["balanced_abstention_acc"],
            "decision": e["decision"], "reason": e["reason"],
        } for e in entries]).style.format(precision=3, subset=["R@5", "P@5", "balanced abstention"]),
            hide_index=True, width="stretch")


def about_page() -> None:
    st.markdown(
        """
**What it does.** Finds the contract clauses that answer a question and returns each passage with
its source contract and exact character offsets. When no passage is relevant enough, it says so
instead of guessing.

**How the final pipeline works**
1. Identify the contract the question names (knowledge-graph party and alias matching), or use
   the contract you pick.
2. Rank that contract's clauses with question-only embeddings and BM25, fused with RRF.
3. Rerank the top 40 with a cross-encoder, then abstain if the best score is below a threshold
   calibrated on training queries.
4. Optionally, Gemini reads the top passages and returns a short answer whose quote is checked
   word for word against the contract.

**Privacy.** Retrieval runs locally. Only the optional Gemini answer sends the top passages to
Google's API; turn it off (or set `answer.allow_external: false`) for confidential contracts.
Logs never contain query or contract text.

**Limitations.** Abstention is not validated on the test split; "no answer found" does not prove a
clause is absent. Answers can miss qualifications stated elsewhere in the contract. Always verify
the highlighted text. See `REPORT.md` and `docs/ARCHITECTURE.md`.
"""
    )


def main() -> None:
    st.set_page_config(page_title="Contract Intelligence", page_icon="📄", layout="wide")
    st.title("Contract Intelligence Assistant")
    st.caption("Clause search over 29 CUAD contracts: ranked passages with exact offsets, honest abstention, "
               "and optional grounded answers.")

    configs = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "configs").glob("*.yaml"))
    with st.sidebar:
        st.header("Settings")
        default = "configs/final.yaml" if "configs/final.yaml" in configs else configs[0]
        config_path = st.selectbox("Pipeline", configs, index=configs.index(default),
                                   help="configs/final.yaml is the configuration selected on train.")
        base = load_config(PROJECT_ROOT / config_path)
        top_k = st.slider("Passages to return", 1, 20, base.retrieval.top_k)

        load_dotenv()
        has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        generate = st.toggle("Precise answer with Gemini", value=has_key, disabled=not has_key,
                             help="Sends the top passages (contract text) to Google's Gemini API. The "
                                  "answer must quote a retrieved passage verbatim.")
        if not has_key:
            st.caption("Set GEMINI_API_KEY in a git-ignored .env file to enable answers.")

        with st.expander("Advanced"):
            abstain = st.checkbox("Abstain when nothing is relevant enough", value=True)
            threshold = st.number_input(
                f"Abstention threshold ({SCORE_NAMES.get(base.retrieval.pipeline, 'score')})",
                value=float(base.retrieval.abstain_threshold or 0.0), step=0.01, format="%.4f",
                disabled=not abstain,
                help="Calibrated on train queries for this config. A relevance score, not a probability.",
            )
            model = st.text_input("Gemini model", AnswerConfig().model, disabled=not generate)

        st.markdown("**Examples**")
        for example in EXAMPLES:
            if st.button(example[:64] + ("..." if len(example) > 64 else ""), key=example,
                         width="stretch"):
                st.session_state["question"] = example

    search, evaluation, about = st.tabs(["Search", "Evaluation", "About"])
    with search:
        search_page(config_path, base, top_k, abstain, threshold, generate, model)
    with evaluation:
        evaluation_page()
    with about:
        about_page()


main()
