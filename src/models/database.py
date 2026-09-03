from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  ticker TEXT NOT NULL,
  form TEXT NOT NULL,
  query TEXT NOT NULL,
  route TEXT,
  latency_ms REAL,
  model TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  estimated_cost REAL,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_runs_created_at ON query_runs(created_at);
"""


class QueryRunStore(Protocol):
    """Storage boundary implemented by SQLite locally and replaceable by PostgreSQL."""

    def log_query(self, **kwargs) -> None: ...

    def latency_stats(self) -> dict[str, float | int]: ...


class Database:
    """Local SQLite metadata/experiment store; interface can be replaced by PostgreSQL later."""

    def __init__(self, path: Path | None = None):
        self.path = path or get_settings().sqlite_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_query(
        self,
        *,
        request_id: str,
        ticker: str,
        form: str,
        query: str,
        route: str,
        latency_ms: float,
        model: str = '',
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO query_runs
                (request_id, created_at, ticker, form, query, route, latency_ms, model, input_tokens, output_tokens, estimated_cost, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    datetime.now(UTC).isoformat(),
                    ticker,
                    form,
                    query,
                    route,
                    latency_ms,
                    model,
                    input_tokens,
                    output_tokens,
                    estimated_cost,
                    json.dumps(metadata or {}),
                ),
            )

    def latency_stats(self) -> dict[str, float | int]:
        with self.connect() as conn:
            values = [
                float(row['latency_ms'])
                for row in conn.execute('SELECT latency_ms FROM query_runs WHERE latency_ms IS NOT NULL')
            ]
        if not values:
            return {'count': 0, 'average_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}
        values.sort()

        def pct(p: float) -> float:
            idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
            return values[idx]

        return {
            'count': len(values),
            'average_ms': sum(values) / len(values),
            'p50_ms': pct(0.5),
            'p95_ms': pct(0.95),
        }
