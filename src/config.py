from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'SEC Filing Research & Financial Intelligence Copilot'
    environment: Literal['development', 'test', 'production'] = 'development'
    data_dir: Path = Path('data')
    sec_user_agent: str = Field(default='SEC-RAG-Research/0.1 contact@example.com')
    sec_timeout_seconds: float = 30.0
    sec_max_retries: int = 4
    sec_requests_per_second: float = 8.0
    sec_cache_ttl_seconds: int = 3600

    chunk_target_tokens: int = 700
    chunk_overlap_tokens: int = 80
    chunk_min_tokens: int = 120

    embedding_model: str = 'BAAI/bge-small-en-v1.5'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    retrieval_top_k: int = 20
    rerank_top_k: int = 6
    reranker_enabled: bool = True

    llm_provider: Literal['mock', 'openai_compatible'] = 'mock'
    llm_base_url: str = 'https://api.openai.com/v1'
    llm_api_key: str | None = None
    llm_model: str = 'gpt-4.1-mini'
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    demo_mode: bool = True

    api_base_url: str = 'http://127.0.0.1:8000'
    max_query_chars: int = 5000
    max_tool_calls: int = 5
    request_size_limit_bytes: int = 1_000_000
    sqlite_path: Path = Path('data/metadata/app.db')

    pricing_input_per_1m: float | None = None
    pricing_output_per_1m: float | None = None

    @field_validator('sec_requests_per_second')
    @classmethod
    def _sec_rate(cls, value: float) -> float:
        if not 0 < value <= 10:
            raise ValueError('SEC_REQUESTS_PER_SECOND must be > 0 and <= 10')
        return value

    @field_validator('chunk_overlap_tokens')
    @classmethod
    def _overlap(cls, value: int, info):
        target = info.data.get('chunk_target_tokens', 700)
        if value < 0 or value >= target:
            raise ValueError('CHUNK_OVERLAP_TOKENS must be >= 0 and less than CHUNK_TARGET_TOKENS')
        return value

    def ensure_directories(self) -> None:
        for name in ('raw', 'processed', 'indexes', 'metadata', 'cache'):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
