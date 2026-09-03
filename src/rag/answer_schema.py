from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    source_url: str
    section: str
    accession_number: str = ''
    form: str = ''
    filing_date: str = ''
    snippet: str = ''


class KPI(BaseModel):
    name: str
    value: float | str
    unit: str
    period: str
    source: str
    previous_value: float | str | None = None
    change: float | str | None = None


class ResearchAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    kpis: list[KPI] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    demo_mode: bool = False
