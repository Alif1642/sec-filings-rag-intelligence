from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SECClientError(RuntimeError):
    """Base exception for SEC data access failures."""


class SECRateLimitError(SECClientError):
    """Raised when SEC repeatedly responds with HTTP 429."""


class SECNotFoundError(SECClientError):
    """Raised when a requested SEC resource cannot be found."""


class InvalidSECURL(SECClientError):
    """Raised when a URL is outside approved SEC hosts."""


class SECClient:
    """HTTP client for public SEC EDGAR data with caching and fair-access throttling."""

    ALLOWED_HOSTS = {'www.sec.gov', 'data.sec.gov'}
    TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
    SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'
    COMPANY_FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    COMPANY_CONCEPT_URL = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json'

    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.cache_dir = self.settings.data_dir / 'cache' / 'sec'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or self._build_session()
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.settings.sec_max_retries,
            connect=self.settings.sec_max_retries,
            read=self.settings.sec_max_retries,
            status=self.settings.sec_max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({'GET'}),
            backoff_factor=0.6,
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': self.settings.sec_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'application/json,text/html,application/xhtml+xml,*/*;q=0.8',
        })
        return session

    @staticmethod
    def normalize_cik(cik: str | int) -> str:
        digits = ''.join(ch for ch in str(cik) if ch.isdigit())
        if not digits or len(digits) > 10:
            raise ValueError(f'Invalid CIK: {cik!r}')
        return digits.zfill(10)

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in self.ALLOWED_HOSTS:
            raise InvalidSECURL(f'Only HTTPS SEC hosts are allowed: {url}')
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            raise InvalidSECURL(f'Unsafe SEC URL: {url}')

    def _cache_path(self, url: str, kind: str) -> Path:
        digest = hashlib.sha256(url.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{digest}.{kind}'

    def _read_cache(self, path: Path) -> bytes | None:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.settings.sec_cache_ttl_seconds:
            return None
        return path.read_bytes()

    def _throttle(self) -> None:
        min_interval = 1.0 / self.settings.sec_requests_per_second
        with self._lock:
            now = time.monotonic()
            wait = min_interval - (now - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _get_bytes(self, url: str, *, use_cache: bool = True) -> bytes:
        self._validate_url(url)
        cache_path = self._cache_path(url, 'bin')
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.debug('SEC cache hit', extra={'url': url})
                return cached
        self._throttle()
        started = time.perf_counter()
        try:
            response = self.session.get(url, timeout=self.settings.sec_timeout_seconds)
        except requests.RequestException as exc:
            raise SECClientError(f'SEC request failed: {exc}') from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info('SEC request', extra={'url': url, 'status_code': response.status_code, 'latency_ms': round(elapsed_ms, 2)})
        if response.status_code == 429:
            raise SECRateLimitError('SEC rate limit reached after retries. Reduce request rate and retry later.')
        if response.status_code == 404:
            raise SECNotFoundError(f'SEC resource not found: {url}')
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise SECClientError(f'SEC HTTP {response.status_code}: {url}') from exc
        cache_path.write_bytes(response.content)
        return response.content

    def get_json(self, url: str, *, use_cache: bool = True) -> dict[str, Any]:
        raw = self._get_bytes(url, use_cache=use_cache)
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SECClientError(f'Invalid JSON returned by SEC: {url}') from exc
        if not isinstance(value, dict):
            raise SECClientError(f'Unexpected SEC JSON payload type: {type(value).__name__}')
        return value

    def get_text(self, url: str, *, use_cache: bool = True) -> str:
        return self._get_bytes(url, use_cache=use_cache).decode('utf-8', errors='replace')

    def get_company_submissions(self, cik: str | int) -> dict[str, Any]:
        return self.get_json(self.SUBMISSIONS_URL.format(cik=self.normalize_cik(cik)))

    def get_company_facts(self, cik: str | int) -> dict[str, Any]:
        return self.get_json(self.COMPANY_FACTS_URL.format(cik=self.normalize_cik(cik)))

    def get_company_concept(self, cik: str | int, taxonomy: str, concept: str) -> dict[str, Any]:
        if not taxonomy.replace('-', '').isalnum() or not concept.replace('_', '').isalnum():
            raise ValueError('Unsafe XBRL taxonomy or concept')
        return self.get_json(self.COMPANY_CONCEPT_URL.format(cik=self.normalize_cik(cik), taxonomy=taxonomy, concept=concept))

    def resolve_ticker(self, ticker: str) -> dict[str, Any]:
        ticker = ticker.strip().upper()
        if not ticker or len(ticker) > 12 or not all(ch.isalnum() or ch in '.-' for ch in ticker):
            raise ValueError('Invalid ticker')
        payload = self.get_json(self.TICKERS_URL)
        for row in payload.values():
            if str(row.get('ticker', '')).upper() == ticker:
                cik = self.normalize_cik(row['cik_str'])
                return {'ticker': ticker, 'cik': cik, 'title': row.get('title', '')}
        raise SECNotFoundError(f'Ticker not found in SEC mapping: {ticker}')

    def get_filing_metadata(self, cik: str | int, form: str = '10-K', filing_date: str | None = None) -> dict[str, Any]:
        cik10 = self.normalize_cik(cik)
        submissions = self.get_company_submissions(cik10)
        recent = submissions.get('filings', {}).get('recent', {})
        keys = ('accessionNumber', 'filingDate', 'reportDate', 'form', 'primaryDocument', 'primaryDocDescription')
        columns = {key: recent.get(key, []) for key in keys}
        count = len(columns['accessionNumber'])
        candidates: list[dict[str, Any]] = []
        for idx in range(count):
            row = {key: (columns[key][idx] if idx < len(columns[key]) else '') for key in keys}
            if row['form'] != form:
                continue
            if filing_date and row['filingDate'] != filing_date:
                continue
            accession = row['accessionNumber']
            accession_nodash = accession.replace('-', '')
            cik_nolead = str(int(cik10))
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{row['primaryDocument']}"
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodash}/{accession}-index.htm"
            row.update({'cik': cik10, 'filing_url': filing_url, 'index_url': index_url})
            candidates.append(row)
        if not candidates:
            suffix = f' on {filing_date}' if filing_date else ''
            raise SECNotFoundError(f'No {form} filing found for CIK {cik10}{suffix}')
        candidates.sort(key=lambda x: x['filingDate'], reverse=True)
        return candidates[0]

    def list_filings(self, cik: str | int, form: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        submissions = self.get_company_submissions(cik)
        recent = submissions.get('filings', {}).get('recent', {})
        rows = []
        for idx, accession in enumerate(recent.get('accessionNumber', [])):
            row_form = recent.get('form', [''])[idx]
            if form and row_form != form:
                continue
            rows.append({
                'accession_number': accession,
                'filing_date': recent.get('filingDate', [''])[idx],
                'report_date': recent.get('reportDate', [''])[idx],
                'form': row_form,
                'primary_document': recent.get('primaryDocument', [''])[idx],
            })
            if len(rows) >= limit:
                break
        return rows

    def download_filing(self, filing_url: str) -> str:
        return self.get_text(filing_url)
