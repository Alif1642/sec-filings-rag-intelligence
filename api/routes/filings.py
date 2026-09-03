from fastapi import APIRouter, HTTPException

from ingestion.pipeline import IngestionPipeline
from ingestion.sec_client import SECClientError
from src.models.schemas import IngestRequest

router = APIRouter(tags=['filings'])


@router.post('/ingest')
def ingest(request: IngestRequest) -> dict:
    try:
        return IngestionPipeline().ingest(request.ticker, request.form, request.filing_date)
    except (ValueError, SECClientError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
