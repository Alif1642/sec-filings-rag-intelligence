from __future__ import annotations

from ingestion.sec_client import SECClient
from src.financials.companyfacts import CompanyFactsService


class XBRLTool:
    def __init__(self, client: SECClient | None = None):
        self.client = client or SECClient()
        self.facts = CompanyFactsService(self.client)

    def get_company_facts(self, ticker: str) -> dict:
        company = self.client.resolve_ticker(ticker)
        return self.client.get_company_facts(company['cik'])

    def get_company_metric(self, ticker: str, metric: str, fiscal_year: int | None = None, form: str = '10-K') -> dict:
        company = self.client.resolve_ticker(ticker)
        fact = self.facts.get_metric(company['cik'], metric, fiscal_year=fiscal_year, form=form)
        result = fact.to_dict()
        result.update({'ticker': company['ticker'], 'company_name': company['title']})
        return result
