from __future__ import annotations

from ingestion.sec_client import SECClient


class CompanyResolver:
    """Resolve public-company tickers using the SEC's official ticker/CIK mapping."""

    def __init__(self, client: SECClient | None = None):
        self.client = client or SECClient()

    def resolve(self, ticker: str) -> dict:
        return self.client.resolve_ticker(ticker)
