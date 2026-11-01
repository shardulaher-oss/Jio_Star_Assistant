"""Rule-based extraction of contract structure and entities.

Every extracted item carries (start, end) offsets into the original document text,
so each graph node and edge can be traced back to the exact words that justify it.
Nothing is inferred beyond what a pattern matched: an unmatched relationship is
absent from the graph, never guessed.

Patterns were written from general contract-drafting conventions, not from the
benchmark's gold spans. They favour precision over recall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contract_intelligence.indexing.segmentation import segment


@dataclass(frozen=True)
class Extract:
    kind: str  # section | definition | reference | party | date | right | restriction | ip
    start: int
    end: int
    text: str  # the matched evidence, verbatim
    key: str = ""  # normalised identifier: section number, defined term, party name, ...
    attrs: tuple[tuple[str, str], ...] = ()


# --- Sections ---------------------------------------------------------------------

_HEADING = re.compile(
    r"(?:^|\n[ \t]*\n|[ \t]{3,})[ \t]*"
    r"(?P<num>(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLC]+(?:\.\d+)*|\d{1,2}(?:\.\d{1,2}){0,3})"
    r"[.:)]?[ \t]+(?P<title>[A-Z][^\n.;:]{0,80})"
)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(value: str) -> int | None:
    if not value or any(ch not in _ROMAN for ch in value):
        return None
    total = 0
    for ch, nxt in zip(value, value[1:] + " "):
        v = _ROMAN[ch]
        total += -v if nxt in _ROMAN and _ROMAN[nxt] > v else v
    return total


def normalise_section_number(raw: str) -> str:
    """'Section 4.2' -> '4.2', 'ARTICLE V' -> '5', '9.2.' -> '9.2'."""
    value = re.sub(r"^(?:article|section|clause|paragraph)s?\s+", "", raw.strip(), flags=re.I)
    value = value.rstrip(".:)")
    roman = _roman_to_int(value.upper()) if value.isalpha() else None
    return str(roman) if roman else value


def extract_sections(text: str) -> list[Extract]:
    sections = []
    for match in _HEADING.finditer(text):
        number = normalise_section_number(match.group("num"))
        if not number or not re.fullmatch(r"\d+(\.\d+)*", number) or int(number.split(".")[0]) > 60:
            continue  # page numbers, years and other noise
        title = match.group("title").strip()
        sections.append(Extract(
            "section", match.start("num"), match.end("title"),
            text[match.start("num") : match.end("title")], key=number,
            attrs=(("title", title[:80]),),
        ))
    return sections


# --- Definitions ------------------------------------------------------------------

_Q_OPEN = "\"“'‘"
_Q_CLOSE = "\"”'’"
_TERM = rf"[{_Q_OPEN}]{{1,2}}(?P<term>[A-Z][A-Za-z0-9&/\-' ]{{1,50}}?)[{_Q_CLOSE}]{{1,2}}"
_EXPLICIT_DEF = re.compile(
    _TERM + r"\s*(?:,\s*)?(?:shall\s+)?(?:mean|means|shall\s+have\s+the\s+meaning|has\s+the\s+meaning|refers?\s+to|shall\s+refer\s+to|includes?)\b"
)
_ALIAS_DEF = re.compile(
    r"\((?:hereinafter\s+(?:referred\s+to\s+as\s+)?|the\s+|each\s+a\s+|each,?\s+an?\s+|collectively,?\s+(?:the\s+)?|individually,?\s+(?:a\s+|the\s+)?)?"
    + _TERM + r"\s*(?:or\s+[^)]{0,60})?\)"
)


def extract_definitions(text: str) -> list[Extract]:
    """Explicit ("X" means ...) and parenthetical alias ((the "X")) definitions.

    The evidence span is the whole sentence or paragraph unit containing the match.
    """
    units = segment(text)
    found: dict[str, Extract] = {}
    for pattern, style in ((_EXPLICIT_DEF, "explicit"), (_ALIAS_DEF, "alias")):
        for match in pattern.finditer(text):
            term = " ".join(match.group("term").split())
            if len(term) < 2 or term.lower() in {"a", "an", "the"}:
                continue
            start, end = next(((a, b) for a, b in units if a <= match.start() < b), (match.start(), match.end()))
            current = found.get(term)
            # Keep the first explicit definition; an alias only if nothing explicit exists.
            if current is None or (style == "explicit" and dict(current.attrs)["style"] == "alias"):
                found[term] = Extract("definition", start, end, text[start:end], key=term,
                                      attrs=(("style", style),))
    return sorted(found.values(), key=lambda e: e.start)


# --- Cross-references -------------------------------------------------------------

_REFERENCE = re.compile(
    r"\b(?P<label>Sections?|Clauses?|Articles?|Paragraphs?|sections?|clauses?|articles?|paragraphs?)\s+"
    r"(?P<num>\d{1,2}(?:\.\d{1,2}){0,3}|[IVXLC]{1,6})\b(?:\s*\((?P<sub>[a-z0-9]{1,4})\))?"
)


def extract_references(text: str) -> list[Extract]:
    refs = []
    for match in _REFERENCE.finditer(text):
        number = normalise_section_number(match.group("num"))
        if not re.fullmatch(r"\d+(\.\d+)*", number):
            continue
        refs.append(Extract("reference", match.start(), match.end(), match.group(0), key=number,
                            attrs=(("label", match.group("label").lower().rstrip("s")),)))
    return refs


# --- Parties ----------------------------------------------------------------------

# Name tokens may end in "." only when they are initials ("R. C. Boyd"), so a name
# cannot run across a sentence end; the suffix must end at a word boundary so "AG"
# does not match the start of "AGREEMENT".
_ORG = re.compile(
    r"\b(?P<name>(?:(?:[A-Z]\.|[A-Z][A-Za-z0-9&'\-]*,?)\s+){0,6}?[A-Z][A-Za-z0-9&'\-]*,?\s+"
    r"(?:Inc\.|Incorporated|LLC|L\.L\.C\.|Ltd\.?|Limited|Corporation|Corp\.|Company|plc|PLC|LP|L\.P\.|AG|GmbH|S\.A\.|N\.V\.|B\.V\.|FSB)"
    r"(?![A-Za-z]))"
)
PREAMBLE_CHARS = 3000


def extract_parties(text: str) -> list[Extract]:
    """Organisation names in the preamble (first PREAMBLE_CHARS characters)."""
    parties: dict[str, Extract] = {}
    for match in _ORG.finditer(text[:PREAMBLE_CHARS]):
        name = " ".join(match.group("name").replace(",", " ").split())
        while name.lower().startswith(("this ", "the ", "and ", "between ", "by ")):
            name = name.split(" ", 1)[1]
        if len(name.split()) >= 2 and name not in parties:
            parties[name] = Extract("party", match.start("name"), match.end("name"),
                                    match.group("name"), key=name)
    return list(parties.values())


# --- Dates, rights, restrictions, IP ownership (sentence level) --------------------

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = re.compile(rf"\b(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTH},?\s+\d{{4}})\b")
_EXPIRY_CUE = re.compile(r"\b(expir\w*|terminat\w*|until|end(?:s|ing)?\s+on|continue\s+in\s+(?:full\s+)?force|term\s+of\s+this)\b", re.I)
_GRANT = re.compile(r"\bgrants?\b.{0,200}?\b(licen[cs]e|sublicen[cs]e|right)", re.I | re.S)
_RESTRICT = re.compile(r"\b(shall\s+not|may\s+not|will\s+not|must\s+not|is\s+prohibited|are\s+prohibited|without\s+the\s+prior\s+written\s+consent)\b", re.I)
_IP_CUE = re.compile(r"\b(intellectual\s+property|copyright|patent|trademark|trade\s+mark|work\s+product|derivative|invention|know-how|proprietary)\b", re.I)
_OWNED = re.compile(
    r"(?:(?:shall\s+(?:be|remain|become)|is|are)\s+(?:and\s+shall\s+remain\s+)?(?:the\s+)?(?:sole\s+and\s+exclusive\s+|exclusive\s+|sole\s+)?property\s+of"
    r"|shall\s+(?:solely\s+)?own|owned\s+(?:solely\s+|exclusively\s+)?by|vest(?:s|ed)?\s+(?:solely\s+)?in)"
    r"\s+(?P<owner>(?:the\s+)?[A-Z][A-Za-z0-9&\-']*(?:\s+[A-Z][A-Za-z0-9&\-']*){0,4})",
)
_RIGHT_FLAGS = ("exclusive", "non-exclusive", "nonexclusive", "perpetual", "irrevocable",
                "royalty-free", "worldwide", "non-transferable", "transferable", "sublicensable")


def _topic(sentence: str) -> str:
    lowered = sentence.lower()
    for topic, cue in (("sublicense", "sublicen"), ("assignment", "assign"), ("competition", "compet"),
                       ("solicitation", "solicit"), ("disclosure", "disclos"), ("use", " use ")):
        if cue in lowered:
            return topic
    return "other"


def extract_sentence_entities(text: str) -> list[Extract]:
    out: list[Extract] = []
    for start, end in segment(text):
        sentence = text[start:end]
        if _EXPIRY_CUE.search(sentence):
            for match in _DATE.finditer(sentence):
                out.append(Extract("date", start + match.start(), start + match.end(), match.group(0),
                                   key=" ".join(match.group(0).split()),
                                   attrs=(("sentence_start", str(start)), ("sentence_end", str(end)))))
        grant = _GRANT.search(sentence)
        if grant:
            lowered = sentence.lower()
            kind = "sublicense" if "sublicen" in lowered else "license" if "licen" in lowered else "right"
            flags = ",".join(f for f in _RIGHT_FLAGS if re.search(rf"(?<![\w-]){f}(?![\w-])", lowered))
            out.append(Extract("right", start, end, sentence, key=kind, attrs=(("flags", flags),)))
        if _RESTRICT.search(sentence):
            out.append(Extract("restriction", start, end, sentence, key=_topic(sentence)))
        owned = _OWNED.search(sentence)
        if owned and _IP_CUE.search(sentence):
            owner = re.sub(r"^the\s+", "", owned.group("owner"))
            out.append(Extract("ip", start, end, sentence, key=owner))
    return out


def extract_all(text: str) -> list[Extract]:
    return (
        extract_sections(text) + extract_definitions(text) + extract_references(text)
        + extract_parties(text) + extract_sentence_entities(text)
    )
