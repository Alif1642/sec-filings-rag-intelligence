from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.sec_client import SECClient
from src.agents.filing_tool import FilingRetrievalTool
from src.agents.router import QueryType, route_query
from src.config import Settings, get_settings
from src.financials.companyfacts import CompanyFactsService, FinancialFactNotFound
from src.financials.comparison import FinancialComparisonService
from src.rag.answer_schema import KPI, ResearchAnswer

METRIC_PATTERNS = [
    ('operating_cash_flow', r'operating cash flow|cash from operations'),
    ('operating_income', r'operating income'),
    ('gross_profit', r'gross profit'),
    ('net_income', r'net income|net earnings'),
    ('shares_outstanding', r'shares outstanding'),
    ('eps', r'earnings per share|\beps\b'),
    ('liabilities', r'liabilit'),
    ('assets', r'\bassets\b'),
    ('cash', r'\bcash\b'),
    ('revenue', r'revenue|net sales|sales'),
]


RETRIEVAL_METRIC_HINTS = {
    'revenue': 'revenue net sales sales',
    'net_income': 'net income net earnings',
    'operating_income': 'operating income operating profit',
    'gross_profit': 'gross profit gross margin',
    'operating_cash_flow': 'operating cash flow cash from operations',
    'cash': 'cash cash equivalents liquidity',
    'assets': 'assets total assets',
    'liabilities': 'liabilities total liabilities',
    'eps': 'earnings per share EPS',
    'shares_outstanding': 'shares outstanding',
}


def build_mixed_retrieval_question(
    question: str,
    metrics: list[str],
    structured_facts: str = '',
) -> str:
    aliases = ' '.join(
        RETRIEVAL_METRIC_HINTS.get(metric, metric.replace('_', ' '))
        for metric in metrics[:3]
    )

    years = list(dict.fromkeys(
        re.findall(r'\b(?:19|20)\d{2}\b', structured_facts)
    ))
    period_hint = ' '.join(years[:4])

    return (
        f"{question}\n"
        f"SEC filing retrieval focus: {aliases}. "
        f"Periods: {period_hint}. "
        "Prioritize Management's Discussion and Analysis and Results of Operations. "
        "Find direct explanations of year-over-year changes using language such as "
        "increased, decreased, primarily due to, driven by, higher, lower, "
        "partially offset, and segment or product drivers."
    ).strip()


def infer_metrics(question: str) -> list[str]:
    q = question.lower()
    found = [name for name, pattern in METRIC_PATTERNS if re.search(pattern, q)]
    return found or ['revenue']


@dataclass(slots=True)
class ResearchResult:
    answer: ResearchAnswer
    retrieved_passages: list[dict]
    timings_ms: dict[str, float]
    token_usage: dict[str, int]
    route: str


class ResearchAgent:
    """Bounded router/orchestrator; it deliberately avoids open-ended agent loops."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = SECClient(self.settings)
        self.filing_tool = FilingRetrievalTool()
        self.facts = CompanyFactsService(self.client)
        self.compare = FinancialComparisonService(self.facts)

    def run(self, ticker: str, form: str, question: str, filing_date: str | None = None) -> ResearchResult:
        if len(question) > self.settings.max_query_chars:
            raise ValueError('Question exceeds configured maximum length')
        route = route_query(question)
        company = self.client.resolve_ticker(ticker)
        tool_calls = 1
        if route == QueryType.FILING_TEXT:
            rag = self.filing_tool.search_filing(ticker, form, question, filing_date)
            return ResearchResult(rag.answer, rag.retrieved_passages, rag.timings_ms, rag.token_usage, route.value)

        metrics = infer_metrics(question)
        structured: list[str] = []
        kpis: list[KPI] = []
        for metric in metrics[:3]:
            if tool_calls >= self.settings.max_tool_calls:
                break
            try:
                if route in {QueryType.COMPARISON, QueryType.CALCULATION, QueryType.MIXED}:
                    comp = self.compare.latest_two(company['cik'], metric, form=form)
                    tool_calls += 1
                    structured.append(
                        f"{metric}: current FY {comp.current.fiscal_year} = {comp.current.value:g} {comp.current.unit}; "
                        f"previous FY {comp.previous.fiscal_year} = {comp.previous.value:g} {comp.previous.unit}; "
                        f"difference = {comp.difference:g}; YoY growth = {comp.growth_pct:.2f}%. "
                        "Source: SEC Company Facts; calculation: deterministic Python."
                    )
                    kpis.append(KPI(name=metric, value=comp.current.value, previous_value=comp.previous.value,
                                    change=f'{comp.growth_pct:.2f}%', unit=comp.current.unit,
                                    period=str(comp.current.fiscal_year), source='SEC Company Facts + deterministic calculation'))
                else:
                    fact = self.facts.get_metric(company['cik'], metric, form=form)
                    tool_calls += 1
                    structured.append(
                        f"{metric}: FY {fact.fiscal_year} = {fact.value:g} {fact.unit}. "
                        f"XBRL concept: {fact.concept}. Source: SEC Company Facts."
                    )
                    kpis.append(KPI(name=metric, value=fact.value, unit=fact.unit,
                                    period=str(fact.fiscal_year), source='SEC Company Facts'))
            except (FinancialFactNotFound, LookupError) as exc:
                structured.append(f'{metric}: unavailable from the applicable SEC Company Facts data ({exc}).')

        if route in {QueryType.FINANCIAL_FACT, QueryType.COMPARISON, QueryType.CALCULATION} and structured:
            # Deterministic answer path: no LLM call required.
            text = ' '.join(structured)
            answer = ResearchAnswer(answer=text, kpis=kpis, caveats=[], demo_mode=self.settings.demo_mode)
            return ResearchResult(answer, [], {'retrieval': 0.0, 'reranking': 0.0, 'generation': 0.0, 'total': 0.0}, {'input': 0, 'output': 0}, route.value)

        # Mixed questions combine filing evidence with XBRL/calculated values.
        structured_facts = '\n'.join(structured)
        retrieval_question = build_mixed_retrieval_question(
            question,
            metrics,
            structured_facts,
        )

        rag = self.filing_tool.run_rag(
            ticker,
            form,
            question,
            filing_date,
            structured_facts,
            retrieval_question=retrieval_question,
        )
        rag.answer.kpis = kpis
        return ResearchResult(rag.answer, rag.retrieved_passages, rag.timings_ms, rag.token_usage, route.value)
