from __future__ import annotations

import re
from enum import StrEnum


class QueryType(StrEnum):
    FILING_TEXT = 'filing_text_question'
    FINANCIAL_FACT = 'financial_fact_question'
    COMPARISON = 'comparison_question'
    CALCULATION = 'calculation_question'
    MIXED = 'mixed_question'


_METRICS = r'revenue|sales|net income|operating income|gross profit|assets|liabilities|cash|cash flow|eps|earnings per share|shares outstanding'
_CALC = r'percentage|percent|growth|margin|difference|change by|how much did'
_COMPARE = r'compare|versus|vs\.?|between|year over year|yoy|previous year|latest two'
_TEXT = r'why|explain|risk|risks|say about|discuss|describe|strategy|business|competition|outlook'


def route_query(question: str) -> QueryType:
    q = question.lower().strip()
    has_metric = bool(re.search(_METRICS, q))
    has_calc = bool(re.search(_CALC, q))
    has_compare = bool(re.search(_COMPARE, q))
    has_text = bool(re.search(_TEXT, q))
    if has_text and (has_metric or has_calc or has_compare):
        return QueryType.MIXED
    if has_calc:
        return QueryType.CALCULATION
    if has_compare:
        return QueryType.COMPARISON
    if has_metric:
        return QueryType.FINANCIAL_FACT
    return QueryType.FILING_TEXT
