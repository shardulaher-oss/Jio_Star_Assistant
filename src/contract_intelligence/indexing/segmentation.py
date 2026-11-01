"""Split contract text into sentence and paragraph units with original offsets.

Units never rewrite text: each is a (start, end) range into the original document,
trimmed of surrounding whitespace. Boundaries are:
- paragraph breaks: a blank line, or a run of 3+ spaces/tabs (many CUAD files were
  flattened from PDF and use wide spaces where the original had line breaks);
- sentence ends: . ! or ? (optionally followed by a closing quote or bracket), then
  whitespace and an uppercase letter, digit, quote or opening bracket.
A sentence end is ignored after common abbreviations ("Inc.", "No.", "U.S."),
single-letter initials ("R. C. Boyd") and clause numbers ("Section 4.", "9.2.").
"""

from __future__ import annotations

import re

ABBREVIATIONS = {
    "inc", "ltd", "llc", "l.l.c", "co", "corp", "plc", "lp", "l.p", "n.a", "s.a", "b.v", "n.v",
    "no", "nos", "sec", "secs", "art", "arts", "para", "paras", "pp", "ex", "exh", "sch",
    "u.s", "u.k", "e.g", "i.e", "etc", "viz", "cf", "vs", "v", "approx", "dept", "fig",
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "messrs",
}
_PARAGRAPH = re.compile(r"\n[ \t\r\f\v]*\n\s*|[ \t]{3,}")
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9])")
_LAST_TOKEN = re.compile(r"(\S+)$")
_ENUMERATOR = re.compile(r"^\(?[0-9ivxlcIVXLC]+(\.[0-9]+)*\)?$")


def _is_false_sentence_end(text: str, punct_pos: int) -> bool:
    if text[punct_pos] != ".":
        return False
    match = _LAST_TOKEN.search(text, max(0, punct_pos - 40), punct_pos)
    if not match:
        return False
    token = match.group(1).lstrip("\"'(“[").lower()
    return (
        token in ABBREVIATIONS
        or (len(token) == 1 and token.isalpha())  # initials
        or bool(_ENUMERATOR.match(token))  # "4." / "9.2." / "(iv)."
    )


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def segment(text: str) -> list[tuple[int, int]]:
    """Return non-empty (start, end) units covering every non-whitespace character."""
    cuts = {0, len(text)}
    for match in _PARAGRAPH.finditer(text):
        cuts.update((match.start(), match.end()))
    for match in _SENTENCE_END.finditer(text):
        if not _is_false_sentence_end(text, match.start()):
            cuts.add(match.end())
    ordered = sorted(cuts)
    units = []
    for left, right in zip(ordered, ordered[1:]):
        start, end = _trim(text, left, right)
        if start < end:
            units.append((start, end))
    return units


def split_long_unit(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Break an over-long unit into pieces of at most max_chars, preferring whitespace."""
    pieces = []
    while end - start > max_chars:
        cut = start + max_chars
        space = text.rfind(" ", start + max_chars // 2, cut)
        cut = space if space != -1 else cut
        piece = _trim(text, start, cut)
        if piece[0] < piece[1]:
            pieces.append(piece)
        start = cut
    piece = _trim(text, start, end)
    if piece[0] < piece[1]:
        pieces.append(piece)
    return pieces
