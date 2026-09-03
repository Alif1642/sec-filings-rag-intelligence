from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from ingestion.filing_parser import DocumentBlock
from src.config import Settings, get_settings


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    ticker: str
    cik: str
    form: str
    filing_date: str
    report_date: str
    accession_number: str
    section: str
    source_url: str
    text: str
    anchor: str | None = None
    untrusted_instruction_like: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class StructureAwareChunker:
    """Chunk blocks by section and paragraph boundaries with configurable token targets."""

    SHORT_TOC_SECTION_RE = re.compile(r"^Item\s+\d+[A-Z]?$", re.IGNORECASE)

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Conservative dependency-free estimate used only for chunk assembly.
        return max(1, int(len(text.split()) * 1.35))

    def chunk(self, blocks: Iterable[DocumentBlock], metadata: dict) -> list[Chunk]:
        blocks = list(blocks)
        result: list[Chunk] = []
        current: list[DocumentBlock] = []
        current_section: str | None = None
        current_tokens = 0

        def flush(*, keep_overlap: bool) -> None:
            nonlocal current, current_tokens, current_section

            if not current:
                return

            text = "\n\n".join(b.text for b in current).strip()

            if text:
                result.append(
                    self._make_chunk(text, current, metadata, len(result))
                )

            if keep_overlap and self.settings.chunk_overlap_tokens > 0:
                current = self._bounded_overlap(current)
                current_tokens = sum(
                    self.estimate_tokens(b.text) for b in current
                )
                current_section = current[-1].section if current else None
            else:
                current = []
                current_tokens = 0
                current_section = None

        for block in blocks:
            block_tokens = self.estimate_tokens(block.text)

            section_changed = (
                current_section is not None
                and block.section != current_section
            )

            too_large = (
                bool(current)
                and current_tokens + block_tokens
                > self.settings.chunk_target_tokens
            )

            if section_changed:
                # Do not carry overlap across SEC section boundaries.
                flush(keep_overlap=False)

            elif too_large:
                flush(keep_overlap=True)

            if block_tokens > self.settings.chunk_target_tokens:
                for piece in self._split_large_text(block.text):
                    synthetic = DocumentBlock(
                        block.order,
                        block.kind,
                        piece,
                        block.section,
                        block.anchor,
                        block.untrusted_instruction_like,
                    )

                    piece_tokens = self.estimate_tokens(piece)

                    if (
                        current
                        and current_tokens + piece_tokens
                        > self.settings.chunk_target_tokens
                    ):
                        flush(keep_overlap=True)

                    current.append(synthetic)
                    current_tokens += piece_tokens
                    current_section = block.section

            else:
                current.append(block)
                current_tokens += block_tokens
                current_section = block.section

        flush(keep_overlap=False)

        # Remove short Table-of-Contents fragments such as "Item 1A",
        # while preserving genuine short filing sections.
        return [
            chunk
            for chunk in result
            if not self._is_toc_fragment(chunk)
        ]

    def _bounded_overlap(
        self,
        blocks: list[DocumentBlock],
    ) -> list[DocumentBlock]:
        """Build overlap without exceeding the configured token budget."""

        budget = self.settings.chunk_overlap_tokens

        if budget <= 0:
            return []

        overlap: list[DocumentBlock] = []
        remaining = budget

        for block in reversed(blocks):
            if remaining <= 0:
                break

            block_tokens = self.estimate_tokens(block.text)

            if block_tokens <= remaining:
                overlap.append(block)
                remaining -= block_tokens
                continue

            # If the block itself is larger than the remaining overlap
            # budget, preserve only its tail instead of the entire block.
            tail_words = max(1, int(remaining / 1.35))
            tail_text = " ".join(
                block.text.split()[-tail_words:]
            ).strip()

            if tail_text:
                overlap.append(
                    DocumentBlock(
                        block.order,
                        block.kind,
                        tail_text,
                        block.section,
                        block.anchor,
                        block.untrusted_instruction_like,
                    )
                )

            break

        return list(reversed(overlap))

    def _is_toc_fragment(self, chunk: Chunk) -> bool:
        """Identify short Table-of-Contents item fragments."""

        return (
            self.estimate_tokens(chunk.text)
            < self.settings.chunk_min_tokens
            and self.SHORT_TOC_SECTION_RE.fullmatch(
                chunk.section.strip()
            )
            is not None
        )

    def _split_large_text(self, text: str) -> list[str]:
        words = text.split()

        max_words = max(
            50,
            int(self.settings.chunk_target_tokens / 1.35),
        )

        overlap_words = int(
            self.settings.chunk_overlap_tokens / 1.35
        )

        step = max(1, max_words - overlap_words)

        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), step)
        ]

    def _make_chunk(
        self,
        text: str,
        blocks: list[DocumentBlock],
        metadata: dict,
        index: int,
    ) -> Chunk:
        accession = (
            metadata.get("accessionNumber")
            or metadata.get("accession_number", "")
        )

        stable = (
            f"{accession}|{blocks[0].section}|"
            f"{index}|{text[:120]}"
        )

        chunk_id = hashlib.sha1(
            stable.encode("utf-8")
        ).hexdigest()[:16]

        return Chunk(
            chunk_id=chunk_id,
            ticker=metadata.get("ticker", ""),
            cik=metadata.get("cik", ""),
            form=metadata.get("form", ""),
            filing_date=(
                metadata.get("filingDate")
                or metadata.get("filing_date", "")
            ),
            report_date=(
                metadata.get("reportDate")
                or metadata.get("report_date", "")
            ),
            accession_number=accession,
            section=blocks[0].section,
            source_url=metadata.get("filing_url", ""),
            text=text,
            anchor=next(
                (b.anchor for b in blocks if b.anchor),
                None,
            ),
            untrusted_instruction_like=any(
                b.untrusted_instruction_like
                for b in blocks
            ),
        )