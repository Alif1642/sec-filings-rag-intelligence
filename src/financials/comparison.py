from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from src.financials.calculations import difference, yoy_growth
from src.financials.companyfacts import CompanyFactsService, FinancialFact


@dataclass(slots=True)
class PeriodComparison:
    metric: str
    current: FinancialFact
    previous: FinancialFact
    difference: float
    growth_pct: float
    calculation_source: str = (
        'Deterministic Python calculation over SEC XBRL facts'
    )

    def to_dict(self) -> dict:
        return asdict(self)


class FinancialComparisonService:
    def __init__(
        self,
        facts: CompanyFactsService | None = None,
    ):
        self.facts = facts or CompanyFactsService()

    @staticmethod
    def _duration_days(
        fact: FinancialFact,
    ) -> int | None:
        if not fact.start or not fact.end:
            return None

        try:
            return (
                date.fromisoformat(fact.end)
                - date.fromisoformat(fact.start)
            ).days
        except ValueError:
            return None

    def _select_previous(
        self,
        series: list[FinancialFact],
        form: str,
    ) -> FinancialFact:
        current = series[0]
        previous = series[1]

        if form != '10-Q':
            return previous

        current_days = self._duration_days(current)
        previous_days = self._duration_days(previous)

        # Point-in-time balance-sheet facts are safe to compare
        # sequentially.
        if current_days is None or previous_days is None:
            return previous

        # Discrete quarters of approximately the same duration
        # can be compared sequentially.
        if abs(current_days - previous_days) <= 7:
            return previous

        # Cumulative YTD periods such as cash flow Q2/Q3 should
        # instead be compared with the same fiscal period from
        # the prior fiscal year.
        for candidate in series[1:]:
            candidate_days = self._duration_days(candidate)

            if candidate_days is None:
                continue

            if candidate.fiscal_period != current.fiscal_period:
                continue

            if (
                current.fiscal_year is not None
                and candidate.fiscal_year is not None
                and candidate.fiscal_year >= current.fiscal_year
            ):
                continue

            if abs(current_days - candidate_days) <= 7:
                return candidate

        raise LookupError(
            'No comparable prior 10-Q period found for '
            f'{current.metric} {current.fiscal_period}'
        )

    def latest_two(
        self,
        cik: str,
        metric: str,
        form: str = '10-K',
    ) -> PeriodComparison:
        # Quarterly comparisons may need prior-year same-quarter
        # data when the latest SEC facts are cumulative YTD values.
        limit = 12 if form == '10-Q' else 2

        series = self.facts.get_series(
            cik,
            metric,
            form=form,
            limit=limit,
        )

        if len(series) < 2:
            raise LookupError(
                f'Need two periods to compare {metric}'
            )

        current = series[0]
        previous = self._select_previous(
            series,
            form,
        )

        return PeriodComparison(
            metric=metric,
            current=current,
            previous=previous,
            difference=difference(
                current.value,
                previous.value,
            ),
            growth_pct=yoy_growth(
                current.value,
                previous.value,
            ),
        )