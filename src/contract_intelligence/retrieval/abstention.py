"""Abstention and query validation: when the system returns nothing, and why."""

from __future__ import annotations

import unicodedata

from contract_intelligence.config import RetrievalConfig
from contract_intelligence.schemas import Passage

MAX_QUERY_CHARS = 2000


def clean_query(query: str) -> tuple[str, str | None]:
    """Return (cleaned query, rejection reason or None).

    Control characters are removed and whitespace collapsed; an empty or over-long
    query is rejected rather than truncated, so the user sees why nothing came back.
    """
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in query or "")
    text = " ".join(text.split())
    if not text:
        return text, "The query is empty."
    if len(text) > MAX_QUERY_CHARS:
        return text, f"The query is too long ({len(text)} characters; the limit is {MAX_QUERY_CHARS})."
    return text, None


def decide(passages: list[Passage], config: RetrievalConfig) -> tuple[bool, str]:
    """Return (abstain, message) for a ranked candidate list.

    The threshold applies to the top-1 score of the configured pipeline (cosine,
    graph score or cross-encoder logit). It was calibrated on train queries and is a
    relevance cut-off, not a probability.
    """
    if not passages:
        return True, "No passages were retrieved."
    if config.abstain_threshold is not None and passages[0].score < config.abstain_threshold:
        return True, (
            f"No passage is relevant enough: best score {passages[0].score:.3f} is below "
            f"the threshold {config.abstain_threshold:.3f}."
        )
    return False, f"Returned {len(passages)} passages."
