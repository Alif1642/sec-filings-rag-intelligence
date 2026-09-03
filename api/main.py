from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.routes import company, filings, health, metrics, query
from src.config import get_settings
from src.observability.logging import configure_logging

settings = get_settings()
configure_logging()


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > settings.request_size_limit_bytes:
            return JSONResponse({'detail': 'Request body too large'}, status_code=413)
        return await call_next(request)


app = FastAPI(
    title=settings.app_name,
    version='0.1.0',
    description='Citation-grounded research over SEC filings and XBRL Company Facts.',
)
app.add_middleware(RequestSizeLimitMiddleware)
app.include_router(health.router)
app.include_router(company.router)
app.include_router(filings.router)
app.include_router(query.router)
app.include_router(metrics.router)
