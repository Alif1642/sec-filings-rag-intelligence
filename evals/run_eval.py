from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from evals.citation_eval import citation_metrics
from evals.financial_eval import financial_errors
from evals.generation_eval import LexicalGroundingJudge
from evals.retrieval_eval import retrieval_metrics
from ingestion.sec_client import SECClient
from src.agents.research_agent import ResearchAgent
from src.financials.companyfacts import CompanyFactsService


class EvaluationRunner:
    """Run a small real-data evaluation set; financial ground truth is fetched from SEC XBRL."""

    def __init__(self):
        self.client = SECClient()
        self.facts = CompanyFactsService(self.client)
        self.agent = ResearchAgent()
        self.judge = LexicalGroundingJudge()

    def run(self, dataset_path: str | Path = 'evals/datasets/sample_questions.json') -> dict[str, Any]:
        path = Path(dataset_path)
        items = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(items, list):
            raise ValueError('Evaluation dataset must be a JSON list')
        results = []
        latencies = []
        for item in items:
            started = time.perf_counter()
            result = self.agent.run(item['ticker'], item.get('form', '10-K'), item['question'])
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)
            row: dict[str, Any] = {
                'ticker': item['ticker'],
                'question': item['question'],
                'route': result.route,
                'latency_ms': latency,
                'citation': citation_metrics(result.answer.answer, result.answer.citations),
                'generation': self.judge.score(item['question'], result.answer.answer, [p.get('text', '') for p in result.retrieved_passages]),
            }
            if item.get('expected_metric'):
                company = self.client.resolve_ticker(item['ticker'])
                truth = self.facts.get_metric(company['cik'], item['expected_metric'], form=item.get('form', '10-K'))
                predicted = next((k for k in result.answer.kpis if k.name == item['expected_metric']), None)
                row['financial'] = financial_errors(float(predicted.value), truth.value) if predicted else {'missing_prediction': True}
            expected_section = item.get('expected_section_pattern')
            expected_chunk_ids = set(item.get('expected_chunk_ids', []))
            if expected_section:
                all_chunks = self.agent.filing_tool.ensure_chunks(item['ticker'], item.get('form', '10-K'))
                expected_chunk_ids.update(
                    str(chunk.get('chunk_id'))
                    for chunk in all_chunks
                    if expected_section.lower() in str(chunk.get('section', '')).lower()
                )
                row['section_evidence_found'] = any(
                    expected_section.lower() in passage.get('section', '').lower()
                    for passage in result.retrieved_passages
                )
            if expected_chunk_ids:
                retrieved_ids = [str(passage.get('chunk_id')) for passage in result.retrieved_passages]
                row['retrieval'] = retrieval_metrics(retrieved_ids, expected_chunk_ids)
            results.append(row)
        sorted_lat = sorted(latencies)
        p95_idx = min(len(sorted_lat)-1, max(0, round((len(sorted_lat)-1)*0.95))) if sorted_lat else 0
        return {
            'cases': results,
            'system': {
                'count': len(results),
                'average_latency_ms': statistics.mean(latencies) if latencies else 0.0,
                'p50_latency_ms': statistics.median(latencies) if latencies else 0.0,
                'p95_latency_ms': sorted_lat[p95_idx] if sorted_lat else 0.0,
            },
        }
