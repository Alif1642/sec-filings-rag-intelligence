from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ingestion.document_cleaner import clean_text, flag_untrusted_instructions
from ingestion.section_detector import detect_item_heading, normalize_heading


@dataclass(slots=True)
class DocumentBlock:
    order: int
    kind: str
    text: str
    section: str
    anchor: str | None = None
    untrusted_instruction_like: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class FilingParser:
    """Parse SEC HTML into ordered headings, paragraphs and compact table text."""

    DROP_TAGS = {
    'script',
    'style',
    'noscript',
    'svg',
    'ix:header',
    'ix:hidden',
    'ix:references',
    'ix:resources',
}
    HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def parse(self, raw_html: str) -> list[DocumentBlock]:
        with warnings.catch_warnings():
           warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
           soup = BeautifulSoup(raw_html, 'lxml')
        for tag in soup.find_all(self.DROP_TAGS):
            tag.decompose()

        blocks: list[DocumentBlock] = []
        section = 'Document'
        order = 0
        seen: set[tuple[str, str]] = set()
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'table']):
            if element.name == 'div' and element.find(['p', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=False):
                continue
            if element.name == 'table':
                text = self._table_to_text(element)
                kind = 'table'
            else:
                text = clean_text(element.get_text(' ', strip=True))
                kind = 'heading' if element.name in self.HEADING_TAGS else 'paragraph'
            if len(text) < 2:
                continue
            key = (kind, text)
            if key in seen:
                continue
            seen.add(key)

            detected = detect_item_heading(text)
            if detected:
                section = detected
                kind = 'heading'
            elif kind == 'heading':
                heading = normalize_heading(text)
                if 2 < len(heading) <= 180:
                    section = heading

            anchor = element.get('id') or element.get('name')
            blocks.append(DocumentBlock(
                order=order,
                kind=kind,
                text=text,
                section=section,
                anchor=anchor,
                untrusted_instruction_like=flag_untrusted_instructions(text),
            ))
            order += 1
        if not blocks:
            text = clean_text(soup.get_text('\n', strip=True))
            if text:
                blocks.append(DocumentBlock(0, 'paragraph', text, 'Document', None, flag_untrusted_instructions(text)))
        return blocks

    @staticmethod
    def _table_to_text(table) -> str:
        rows: list[str] = []
        for tr in table.find_all('tr'):
            cells = [clean_text(c.get_text(' ', strip=True)) for c in tr.find_all(['th', 'td'])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(' | '.join(cells))
        return clean_text('\n'.join(rows))
