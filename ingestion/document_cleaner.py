from __future__ import annotations

import html
import re

INJECTION_PATTERNS = re.compile(
    r'(?i)(ignore\s+(all\s+)?previous\s+instructions|system\s+prompt|developer\s+message|act\s+as\s+an?\s+)',
)


def clean_text(text: str) -> str:
    """Normalize SEC filing text while preserving semantic paragraph boundaries."""
    text = html.unescape(text).replace('\xa0', ' ')
    text = re.sub(r'[\t\r\f\v]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def flag_untrusted_instructions(text: str) -> bool:
    """Detect instruction-like text so it can be explicitly treated as untrusted evidence."""
    return bool(INJECTION_PATTERNS.search(text))
