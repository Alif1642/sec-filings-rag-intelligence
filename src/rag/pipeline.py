from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import Settings, get_settings
from src.rag.answer_schema import ResearchAnswer
from src.rag.citation import validate_citations
from src.rag.context import build_context
from src.rag.generator import AnswerGenerator
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import FAISSVectorStore


@dataclass(slots=True)
class RAGRun:
    answer: ResearchAnswer
    retrieved_passages: list[dict]
    timings_ms: dict[str, float]
    token_usage: dict[str, int]


class RAGPipeline:
    def __init__(self, chunks: list[dict], settings: Settings | None = None, vector_store: FAISSVectorStore | None = None):
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(chunks, self.settings, vector_store=vector_store)
        self.reranker = CrossEncoderReranker(self.settings)
        self.generator = AnswerGenerator(self.settings)

    @staticmethod
    def _apply_section_preference(
        question: str,
        hits: list,
    ) -> list:
        q = question.lower()

        risk_intent = any(
            phrase in q
            for phrase in (
                'risk factor',
                'risk factors',
                'major risk',
                'major risks',
                'business risk',
                'business risks',
            )
        )

        if not risk_intent:
            return hits

        item_1a_hits = [
            hit
            for hit in hits
            if str(
                hit.chunk.get('section', '')
            ).lower().startswith('item 1a')
        ]

        # Only enforce the section preference when retrieval
        # already found enough Item 1A evidence.
        if len(item_1a_hits) >= 3:
            return item_1a_hits

        return hits

    def run(
        self,
        question: str,
        structured_facts: str = '',
        retrieval_question: str | None = None,
    ) -> RAGRun:
        search_query = retrieval_question or question

        t0 = time.perf_counter()
        candidates = self.retriever.search(search_query)
        t1 = time.perf_counter()
        hits = self.reranker.rerank(search_query, candidates)
        hits = self._apply_section_preference(question, hits)
        t2 = time.perf_counter()
        context = build_context(hits)
        generated = self.generator.generate(question, context.text, structured_facts)
        t3 = time.perf_counter()
        ok, invalid = validate_citations(generated.text, context.citations)
        caveats = [] if ok else [f'Generator referenced invalid citation IDs: {", ".join(invalid)}']
        answer = ResearchAnswer(
            answer=generated.text,
            citations=context.citations,
            caveats=caveats,
            demo_mode=self.settings.demo_mode,
        )
        return RAGRun(
            answer=answer,
            retrieved_passages=[h.chunk for h in hits],
            timings_ms={
                'retrieval': round((t1-t0)*1000, 2),
                'reranking': round((t2-t1)*1000, 2),
                'generation': round((t3-t2)*1000, 2),
                'total': round((t3-t0)*1000, 2),
            },
            token_usage={'input': generated.input_tokens, 'output': generated.output_tokens},
        )
