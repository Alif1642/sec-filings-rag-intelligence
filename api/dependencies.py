from __future__ import annotations

from functools import lru_cache

from ingestion.sec_client import SECClient
from src.agents.research_agent import ResearchAgent
from src.financials.companyfacts import CompanyFactsService
from src.financials.comparison import FinancialComparisonService
from src.models.database import Database


@lru_cache(maxsize=1)
def get_sec_client() -> SECClient:
    return SECClient()


@lru_cache(maxsize=1)
def get_research_agent() -> ResearchAgent:
    return ResearchAgent()


@lru_cache(maxsize=1)
def get_company_facts_service() -> CompanyFactsService:
    return CompanyFactsService(get_sec_client())


@lru_cache(maxsize=1)
def get_comparison_service() -> FinancialComparisonService:
    return FinancialComparisonService(get_company_facts_service())


@lru_cache(maxsize=1)
def get_database() -> Database:
    return Database()
