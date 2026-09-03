from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.rag.answer_schema import KPI, Citation


class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    form: Literal['10-K', '10-Q'] = '10-K'
    filing_date: str | None = None

    @field_validator('ticker')
    @classmethod
    def upper_ticker(cls, value: str) -> str:
        return value.strip().upper()


class QueryRequest(IngestRequest):
    question: str = Field(min_length=2, max_length=5000)


class FinancialQueryRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    metric: str
    fiscal_year: int | None = None
    form: Literal['10-K', '10-Q'] = '10-K'


class CompareRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    metric: str
    form: Literal['10-K', '10-Q'] = '10-K'


class EvaluateRequest(BaseModel):
    dataset_path: str = 'evals/datasets/sample_questions.json'


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    kpis: list[KPI] = Field(default_factory=list)
    retrieved_passages: list[dict] = Field(default_factory=list)
    latency_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    estimated_cost: float | None = None
    route: str = ''
    caveats: list[str] = Field(default_factory=list)
    demo_mode: bool = False
