from __future__ import annotations

from dataclasses import dataclass

from src.rag.answer_schema import Citation


@dataclass(slots=True)
class BuiltContext:
    text: str
    citations: list[Citation]


def build_context(hits: list, max_chars: int = 24000) -> BuiltContext:
    """Assign stable display citations and build bounded LLM context."""
    parts: list[str] = []
    citations: list[Citation] = []
    used = 0
    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk if hasattr(hit, 'chunk') else hit
        text = str(chunk.get('text', '')).strip()
        if not text:
            continue
        section = str(chunk.get('section', 'Document'))
        header = f"[{idx}] Section: {section}\n"
        payload = header + text
        if used + len(payload) > max_chars and parts:
            break
        parts.append(payload[: max_chars - used])
        used += len(payload)
        citations.append(Citation(
            citation_id=str(idx),
            chunk_id=str(chunk.get('chunk_id', '')),
            source_url=str(chunk.get('source_url', '')),
            section=section,
            accession_number=str(chunk.get('accession_number', '')),
            form=str(chunk.get('form', '')),
            filing_date=str(chunk.get('filing_date', '')),
            snippet=text[:500],
        ))
    return BuiltContext('\n\n'.join(parts), citations)
