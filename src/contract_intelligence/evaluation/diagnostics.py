"""Label each evaluated query with the reason it failed (or that it succeeded)."""

from __future__ import annotations

from collections.abc import Sequence

from contract_intelligence.evaluation.metrics import Span, recall_merged

# Answerable queries
WRONG_ABSTENTION = "wrong_abstention"  # system returned nothing
WRONG_DOCUMENT = "wrong_document"  # no top-k passage comes from a gold contract
WRONG_SECTION = "right_document_wrong_section"  # gold contract retrieved, no gold overlap
PARTIAL_COVERAGE = "partial_coverage"  # some, but not all, gold characters retrieved
FULL_COVERAGE = "full_coverage"
# Unanswerable queries
CORRECT_ABSTENTION = "correct_abstention"
FALSE_ANSWER = "false_answer"  # returned passages for a question with no answer


def diagnose_answerable(retrieved: Sequence[Span], gold: Sequence[Span], abstained: bool) -> str:
    if abstained:
        return WRONG_ABSTENTION
    gold_files = {g.file_path for g in gold}
    if not any(r.file_path in gold_files for r in retrieved):
        return WRONG_DOCUMENT
    coverage = recall_merged(retrieved, gold)
    if coverage == 0:
        return WRONG_SECTION
    return FULL_COVERAGE if coverage >= 0.99 else PARTIAL_COVERAGE


def diagnose_unanswerable(abstained: bool) -> str:
    return CORRECT_ABSTENTION if abstained else FALSE_ANSWER
