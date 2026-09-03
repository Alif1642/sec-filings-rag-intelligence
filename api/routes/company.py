from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_sec_client
from ingestion.sec_client import SECClient, SECClientError

router = APIRouter(prefix='/companies', tags=['companies'])


@router.get('/{ticker}')
def get_company(ticker: str, client: SECClient = Depends(get_sec_client)) -> dict:  # noqa: B008
    try:
        return client.resolve_ticker(ticker)
    except (ValueError, SECClientError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/{ticker}/filings')
def get_filings(
    ticker: str,
    form: str | None = Query(default=None, pattern='^(10-K|10-Q)$'),
    limit: int = Query(default=20, ge=1, le=100),
    client: SECClient = Depends(get_sec_client),  # noqa: B008
) -> dict:
    try:
        company = client.resolve_ticker(ticker)
        return {'company': company, 'filings': client.list_filings(company['cik'], form=form, limit=limit)}
    except (ValueError, SECClientError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
