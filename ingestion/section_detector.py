from __future__ import annotations

import re

ITEM_RE = re.compile(r'(?i)^\s*(item\s+\d+[a-z]?\.?\s*[-—:.]?\s*.+)$')


def normalize_heading(text: str) -> str:
    return ' '.join(text.split()).strip(' -—:.')


def detect_item_heading(text: str) -> str | None:
    """Recognize common 10-K/10-Q `Item` headings without assuming a fixed company."""
    candidate = normalize_heading(text)
    match = ITEM_RE.match(candidate)
    if not match:
        return None
    if len(candidate) > 180:
        return None
    return candidate
