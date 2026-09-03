from fastapi import APIRouter, Depends

from api.dependencies import get_database
from src.models.database import Database

router = APIRouter(tags=['metrics'])


@router.get('/metrics')
def metrics(db: Database = Depends(get_database)) -> dict:  # noqa: B008
    return {'query_latency': db.latency_stats()}
