from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ingestion.pipeline import IngestionPipeline
from src.config import get_settings
from src.rag.answer_schema import ResearchAnswer
from src.rag.pipeline import RAGPipeline, RAGRun
from src.retrieval.vector_store import FAISSVectorStore

# Bump this value whenever RAG answer/retrieval behavior changes in a way
# that should invalidate previously cached query results.
QUERY_CACHE_VERSION = '3'


class FilingRetrievalTool:
    """Load filing chunks, reuse persistent embeddings and cache identical RAG queries."""

    def __init__(self):
        self.settings = get_settings()
        self.ingestion = IngestionPipeline()
        self._pipeline_cache: dict[str, RAGPipeline] = {}
        self.query_cache_dir = self.settings.data_dir / 'cache' / 'queries'
        self.query_cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_chunks(self, ticker: str, form: str = '10-K', filing_date: str | None = None) -> list[dict]:
        processed = self.settings.data_dir / 'processed' / ticker.upper()
        candidates = sorted(
            processed.glob('*/manifest.json'),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if processed.exists() else []
        for manifest_path in candidates:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            meta = manifest.get('metadata', {})
            if meta.get('form') != form:
                continue
            if filing_date and meta.get('filingDate') != filing_date:
                continue
            chunks_path = Path(manifest['chunks_path'])
            if chunks_path.exists():
                return json.loads(chunks_path.read_text(encoding='utf-8'))
        manifest = self.ingestion.ingest(ticker, form, filing_date)
        return json.loads(Path(manifest['chunks_path']).read_text(encoding='utf-8'))

    @staticmethod
    def _filing_key(chunks: list[dict]) -> str:
        if not chunks:
            return 'empty'
        accession = str(chunks[0].get('accession_number', '')).replace('-', '')
        if accession:
            return accession
        stable = '|'.join(str(chunk.get('chunk_id', '')) for chunk in chunks)
        return hashlib.sha256(stable.encode()).hexdigest()[:24]

    def get_pipeline(self, ticker: str, form: str = '10-K', filing_date: str | None = None) -> tuple[str, RAGPipeline]:
        chunks = self.ensure_chunks(ticker, form, filing_date)
        key = self._filing_key(chunks)
        if key in self._pipeline_cache:
            return key, self._pipeline_cache[key]

        index_dir = self.settings.data_dir / 'indexes' / key
        store = FAISSVectorStore(self.settings)
        if (index_dir / 'index.faiss').exists() and (index_dir / 'chunks.json').exists():
            store.load(index_dir)
            # Prevent stale index reuse if the processed chunk set changed.
            if [c.get('chunk_id') for c in store.chunks] != [c.get('chunk_id') for c in chunks]:
                store.build(chunks)
                store.save(index_dir)
        else:
            store.build(chunks)
            store.save(index_dir)
        pipeline = RAGPipeline(chunks, self.settings, vector_store=store)
        self._pipeline_cache[key] = pipeline
        return key, pipeline

    def _query_cache_path(
        self,
        filing_key: str,
        question: str,
        structured_facts: str,
        retrieval_question: str = '',
    ) -> Path:
        signature = '|'.join([
            QUERY_CACHE_VERSION,
            filing_key,
            question.strip(),
            structured_facts.strip(),
            retrieval_question.strip(),
            self.settings.embedding_model,
            self.settings.reranker_model,
            str(self.settings.reranker_enabled),
            self.settings.llm_provider,
            self.settings.llm_model,
            str(self.settings.demo_mode),
        ])
        digest = hashlib.sha256(signature.encode('utf-8')).hexdigest()
        return self.query_cache_dir / f'{digest}.json'

    @staticmethod
    def _serialize_run(run: RAGRun) -> dict:
        return {
            'answer': run.answer.model_dump(),
            'retrieved_passages': run.retrieved_passages,
            'timings_ms': run.timings_ms,
            'token_usage': run.token_usage,
        }

    @staticmethod
    def _deserialize_run(data: dict) -> RAGRun:
        return RAGRun(
            answer=ResearchAnswer.model_validate(data['answer']),
            retrieved_passages=data.get('retrieved_passages', []),
            timings_ms=data.get('timings_ms', {}),
            token_usage=data.get('token_usage', {}),
        )

    def run_rag(
        self,
        ticker: str,
        form: str,
        question: str,
        filing_date: str | None = None,
        structured_facts: str = '',
        retrieval_question: str | None = None,
    ) -> RAGRun:
        filing_key, pipeline = self.get_pipeline(ticker, form, filing_date)
        cache_path = self._query_cache_path(
            filing_key,
            question,
            structured_facts,
            retrieval_question or '',
        )
        if cache_path.exists():
            try:
                return self._deserialize_run(json.loads(cache_path.read_text(encoding='utf-8')))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
        run = pipeline.run(
            question,
            structured_facts,
            retrieval_question=retrieval_question,
        )
        cache_path.write_text(json.dumps(self._serialize_run(run), indent=2), encoding='utf-8')
        return run

    def search_filing(self, ticker: str, form: str, question: str, filing_date: str | None = None) -> RAGRun:
        return self.run_rag(ticker, form, question, filing_date)
