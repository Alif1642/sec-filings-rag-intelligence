from __future__ import annotations

import json
from pathlib import Path

from ingestion.sec_client import SECClient
from src.config import get_settings


class FilingDownloader:
    """Download only explicitly requested SEC filings and cache them locally."""

    def __init__(self, client: SECClient | None = None):
        self.client = client or SECClient()
        self.raw_dir = get_settings().data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download(self, ticker: str, form: str = '10-K', filing_date: str | None = None) -> tuple[dict, Path]:
        company = self.client.resolve_ticker(ticker)
        meta = self.client.get_filing_metadata(company['cik'], form=form, filing_date=filing_date)
        meta.update({'ticker': company['ticker'], 'company_name': company['title']})
        text = self.client.download_filing(meta['filing_url'])
        accession = meta['accessionNumber'].replace('-', '')
        target_dir = self.raw_dir / company['ticker'] / accession
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(meta['primaryDocument']).suffix or '.html'
        path = target_dir / f"filing{suffix}"
        path.write_text(text, encoding='utf-8')
        (target_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return meta, path
