from __future__ import annotations

from urllib.parse import urlparse

from src.rag.answer_schema import Citation
from src.rag.citation import citation_completeness, cited_ids


def citation_correctness(answer: str, citations: list[Citation]) -> float:
    referenced = cited_ids(answer)
    if not referenced:
        return 1.0 if not citations else 0.0
    valid = {c.citation_id for c in citations}
    return len(referenced & valid) / len(referenced)


def citation_source_validity(citations: list[Citation]) -> float:
    if not citations:
        return 1.0
    valid = 0
    for cite in citations:
        parsed = urlparse(cite.source_url)
        if parsed.scheme == 'https' and parsed.hostname in {'www.sec.gov', 'data.sec.gov'}:
            valid += 1
    return valid / len(citations)


def citation_metrics(answer: str, citations: list[Citation]) -> dict[str, float]:
    return {
        'citation_correctness': citation_correctness(answer, citations),
        'citation_completeness': citation_completeness(answer),
        'citation_source_validity': citation_source_validity(citations),
    }
