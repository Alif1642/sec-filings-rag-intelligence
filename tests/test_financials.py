import pytest

from src.financials.calculations import difference, margin, yoy_growth
from src.financials.companyfacts import CompanyFactsService, FinancialFact
from src.financials.comparison import FinancialComparisonService


class StubClient:
    def get_company_facts(self, cik):
        return {
            'facts': {'us-gaap': {
                'RevenueFromContractWithCustomerExcludingAssessedTax': {
                    'units': {'USD': [
                        {'val': 100.0, 'fy': 2025, 'fp': 'FY', 'form': '10-K', 'filed': '2024-11-01', 'start': '2023-10-01', 'end': '2024-09-30', 'accn': 'a'},
                        {'val': 120.0, 'fy': 2025, 'fp': 'FY', 'form': '10-K', 'filed': '2025-11-01', 'start': '2024-10-01', 'end': '2025-09-30', 'accn': 'b'},
                    ]}
                }
            }}
        }


def test_financial_metric_fallback_and_series():
    service = CompanyFactsService(StubClient())
    latest = service.get_metric('1', 'revenue')
    assert latest.value == 120.0 and latest.concept.startswith('RevenueFrom')
    assert [fact.fiscal_year for fact in service.get_series('1', 'revenue', limit=2)] == [2025, 2024]


def test_financial_calculations():
    assert yoy_growth(120, 100) == pytest.approx(20.0)
    assert margin(25, 100) == pytest.approx(25.0)
    assert difference(120, 100) == 20


class FakeQuarterlyFacts:
    def get_series(self, cik, metric, form="10-K", limit=None):
        return [
            FinancialFact(
                metric="operating_cash_flow",
                value=116.996,
                unit="USD",
                fiscal_year=2026,
                fiscal_period="Q3",
                form="10-Q",
                filed="2026-07-31",
                start="2025-09-28",
                end="2026-06-27",
                accession_number="a",
                concept="NetCashProvidedByUsedInOperatingActivities",
                taxonomy="us-gaap",
                source="test",
            ),
            FinancialFact(
                metric="operating_cash_flow",
                value=82.627,
                unit="USD",
                fiscal_year=2026,
                fiscal_period="Q2",
                form="10-Q",
                filed="2026-05-01",
                start="2025-09-28",
                end="2026-03-28",
                accession_number="b",
                concept="NetCashProvidedByUsedInOperatingActivities",
                taxonomy="us-gaap",
                source="test",
            ),
            FinancialFact(
                metric="operating_cash_flow",
                value=81.754,
                unit="USD",
                fiscal_year=2025,
                fiscal_period="Q3",
                form="10-Q",
                filed="2025-08-01",
                start="2024-09-29",
                end="2025-06-28",
                accession_number="c",
                concept="NetCashProvidedByUsedInOperatingActivities",
                taxonomy="us-gaap",
                source="test",
            ),
        ]


def test_quarterly_ytd_cash_flow_compares_same_period_prior_year():
    service = FinancialComparisonService(FakeQuarterlyFacts())

    result = service.latest_two(
        "0000320193",
        "operating_cash_flow",
        form="10-Q",
    )

    assert result.current.fiscal_year == 2026
    assert result.current.fiscal_period == "Q3"

    assert result.previous.fiscal_year == 2025
    assert result.previous.fiscal_period == "Q3"

    assert round(result.growth_pct, 2) == 43.11
