from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_company_facts_service,
    get_comparison_service,
    get_database,
    get_research_agent,
    get_sec_client,
)
from evals.run_eval import EvaluationRunner
from ingestion.sec_client import SECClient, SECClientError
from src.agents.research_agent import ResearchAgent
from src.financials.companyfacts import CompanyFactsService, FinancialFactNotFound
from src.financials.comparison import FinancialComparisonService
from src.models.database import Database
from src.models.schemas import CompareRequest, EvaluateRequest, FinancialQueryRequest, QueryRequest, QueryResponse
from src.observability.metrics import estimate_cost

router = APIRouter(tags=['research'])
logger = logging.getLogger(__name__)


@router.post('/query', response_model=QueryResponse)
def query(
    request: QueryRequest,
    agent: ResearchAgent = Depends(get_research_agent),  # noqa: B008
    db: Database = Depends(get_database),  # noqa: B008
) -> QueryResponse:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        result = agent.run(request.ticker, request.form, request.question, request.filing_date)
    except (ValueError, SECClientError, FinancialFactNotFound, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency = (time.perf_counter() - started) * 1000
    cost = estimate_cost(result.token_usage.get('input', 0), result.token_usage.get('output', 0))
    db.log_query(
        request_id=request_id,
        ticker=request.ticker,
        form=request.form,
        query=request.question,
        route=result.route,
        latency_ms=latency,
        input_tokens=result.token_usage.get('input', 0),
        output_tokens=result.token_usage.get('output', 0),
        estimated_cost=cost,
        metadata={'timings_ms': result.timings_ms, 'retrieved_chunks': len(result.retrieved_passages)},
    )
    logger.info(
        'research query completed',
        extra={
            'request_id': request_id,
            'ticker': request.ticker,
            'filing': request.form,
            'query': request.question,
            'retrieval_ms': result.timings_ms.get('retrieval', 0.0),
            'reranking_ms': result.timings_ms.get('reranking', 0.0),
            'generation_ms': result.timings_ms.get('generation', 0.0),
            'total_ms': latency,
            'chunks': len(result.retrieved_passages),
            'input_tokens': result.token_usage.get('input', 0),
            'output_tokens': result.token_usage.get('output', 0),
            'estimated_cost': cost,
        },
    )
    return QueryResponse(
        answer=result.answer.answer,
        citations=result.answer.citations,
        kpis=result.answer.kpis,
        caveats=result.answer.caveats,
        retrieved_passages=result.retrieved_passages,
        latency_ms=round(latency, 2),
        timings_ms=result.timings_ms,
        token_usage=result.token_usage,
        estimated_cost=cost,
        route=result.route,
        demo_mode=result.answer.demo_mode,
    )


@router.post('/financials/query')
def financial_query(
    request: FinancialQueryRequest,
    facts: CompanyFactsService = Depends(get_company_facts_service),  # noqa: B008
    client: SECClient = Depends(get_sec_client),  # noqa: B008
) -> dict:
    try:
        company = client.resolve_ticker(request.ticker)
        fact = facts.get_metric(company['cik'], request.metric, request.fiscal_year, request.form)
        return {'company': company, 'fact': fact.to_dict()}
    except (ValueError, SECClientError, FinancialFactNotFound) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/compare')
def compare(
    request: CompareRequest,
    service: FinancialComparisonService = Depends(get_comparison_service),  # noqa: B008
    client: SECClient = Depends(get_sec_client),  # noqa: B008
) -> dict:
    try:
        company = client.resolve_ticker(request.ticker)
        comparison = service.latest_two(company['cik'], request.metric, request.form)
        return {'company': company, 'comparison': comparison.to_dict()}
    except (ValueError, SECClientError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/evaluate')
def evaluate(request: EvaluateRequest) -> dict:
    try:
        return EvaluationRunner().run(request.dataset_path)
    except (ValueError, OSError, SECClientError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
