from fastapi.testclient import TestClient

from api.dependencies import get_database, get_research_agent
from api.main import app
from src.agents.research_agent import ResearchResult
from src.rag.answer_schema import KPI, ResearchAnswer


class StubAgent:
    def run(self, ticker, form, question, filing_date=None):
        answer = ResearchAnswer(
            answer='revenue: FY 2025 = 120 USD. Source: SEC Company Facts.',
            kpis=[KPI(name='revenue', value=120.0, unit='USD', period='2025', source='SEC Company Facts')],
            demo_mode=True,
        )
        return ResearchResult(
            answer=answer,
            retrieved_passages=[],
            timings_ms={'retrieval': 0.0, 'reranking': 0.0, 'generation': 0.0, 'total': 0.0},
            token_usage={'input': 0, 'output': 0},
            route='financial_fact_question',
        )


class StubDatabase:
    def log_query(self, **kwargs):
        self.last = kwargs


def test_health_endpoint():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_query_endpoint_without_live_sec():
    app.dependency_overrides[get_research_agent] = lambda: StubAgent()
    app.dependency_overrides[get_database] = lambda: StubDatabase()
    try:
        client = TestClient(app)
        response = client.post('/query', json={'ticker': 'AAPL', 'form': '10-K', 'question': 'What was revenue?'})
        assert response.status_code == 200
        body = response.json()
        assert body['route'] == 'financial_fact_question'
        assert body['kpis'][0]['name'] == 'revenue'
    finally:
        app.dependency_overrides.clear()
