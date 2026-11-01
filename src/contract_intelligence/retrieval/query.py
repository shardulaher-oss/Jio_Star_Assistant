"""Query processing: separate the contract reference from the question itself."""

from __future__ import annotations

import re
from typing import NamedTuple

# Benchmark queries look like "Consider the <agreement> between <parties>; <question>".
_CONSIDER = re.compile(r"^\s*consider\s+(?P<contract>.+?);\s*(?P<question>.+)$", re.I | re.S)


class ParsedQuery(NamedTuple):
    contract: str  # text naming the contract, or "" when the query names none
    question: str  # what is being asked


def parse_query(query: str) -> ParsedQuery:
    match = _CONSIDER.match(query)
    if match:
        return ParsedQuery(match.group("contract").strip(), match.group("question").strip())
    return ParsedQuery("", query.strip())
