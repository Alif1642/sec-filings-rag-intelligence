from __future__ import annotations

import re

from src.rag.answer_schema import Citation

_CITE_RE = re.compile(r'\[(\d+)\]')


def cited_ids(answer: str) -> set[str]:
    return set(_CITE_RE.findall(answer))


def validate_citations(answer: str, citations: list[Citation]) -> tuple[bool, list[str]]:
    valid = {c.citation_id for c in citations}
    referenced = cited_ids(answer)
    invalid = sorted(referenced - valid)
    return not invalid, invalid


def citation_completeness(answer: str) -> float:
    """Heuristic: fraction of factual-looking sentences containing a bracket citation."""
    normalized = re.sub(r'([.!?])\s+(\[\d+\])', r' \2\1', answer)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', normalized) if s.strip()]
    factual = [s for s in sentences if any(ch.isdigit() for ch in s) or re.search(r'\b(reported|increased|decreased|risk|revenue|income|assets|cash)\b', s, re.I)]
    if not factual:
        return 1.0
    cited = sum(bool(_CITE_RE.search(s)) for s in factual)
    return cited / len(factual)
