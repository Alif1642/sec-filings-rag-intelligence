from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class GenerationJudge(Protocol):
    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float]: ...


@dataclass(slots=True)
class LexicalGroundingJudge:
    """Offline baseline with a RAGAS-compatible metric shape; replaceable by a model-based judge."""

    def score(self, question: str, answer: str, contexts: list[str]) -> dict[str, float]:
        context_terms = set(re.findall(r'[a-z0-9]+', ' '.join(contexts).lower()))
        answer_terms = set(re.findall(r'[a-z0-9]+', answer.lower()))
        question_terms = set(re.findall(r'[a-z0-9]+', question.lower()))
        content_answer = {t for t in answer_terms if len(t) > 3}
        content_question = {t for t in question_terms if len(t) > 3}
        faithfulness = len(content_answer & context_terms) / len(content_answer) if content_answer else 1.0
        relevance = len(content_answer & content_question) / len(content_question) if content_question else 1.0
        return {
            'answer_faithfulness': faithfulness,
            'context_precision': faithfulness,
            'context_recall': min(1.0, len(content_question & context_terms) / len(content_question)) if content_question else 1.0,
            'answer_relevance': relevance,
        }
