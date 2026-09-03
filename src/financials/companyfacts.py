from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from ingestion.sec_client import SECClient
from src.financials.metrics import METRIC_CONCEPTS, METRIC_UNITS


class FinancialFactNotFound(LookupError):
    pass


@dataclass(slots=True)
class FinancialFact:
    metric: str
    value: float
    unit: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed: str
    start: str | None
    end: str | None
    accession_number: str | None
    concept: str
    taxonomy: str = 'us-gaap'
    source: str = 'SEC Company Facts'

    def to_dict(self) -> dict:
        return asdict(self)


class CompanyFactsService:
    """Extract normalized financial metrics with concept and taxonomy fallback logic."""

    TAXONOMIES = ('us-gaap', 'dei')

    def __init__(self, client: SECClient | None = None):
        self.client = client or SECClient()

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        compact = unit.replace(' ', '')
        aliases = {
            'USD/shares': 'USD/shares',
            'shares': 'shares',
            'USD': 'USD',
        }
        return aliases.get(compact, unit)

    @staticmethod
    def _normalized_fiscal_year(
        entry: dict,
        form: str,
    ) -> int | None:
        # For 10-Q primary contexts, SEC `fy` is the fiscal year label.
        # This matters for non-calendar fiscal years, where Q1 may end in
        # the previous calendar year (for example FY2026 Q1 ending in 2025).
        if form == '10-Q':
            fy = entry.get('fy')
            if isinstance(fy, int):
                return fy

        # For annual comparative facts, the SEC `fy` value may describe the
        # current filing rather than the comparative period. Period-end year
        # therefore remains the safer normalization for the existing 10-K path.
        end = entry.get('end')
        if isinstance(end, str) and len(end) >= 4 and end[:4].isdigit():
            return int(end[:4])

        fy = entry.get('fy')
        return int(fy) if isinstance(fy, int) else None

    def _iter_metric_facts(self, payload: dict, metric: str, form: str) -> list[FinancialFact]:
        taxonomies = payload.get('facts', {})
        output: list[FinancialFact] = []
        for concept in METRIC_CONCEPTS.get(metric, []):
            for taxonomy in self.TAXONOMIES:
                node = taxonomies.get(taxonomy, {}).get(concept)
                if not node:
                    continue
                for unit, entries in node.get('units', {}).items():
                    normalized_unit = self._normalize_unit(unit)
                    allowed_units = [self._normalize_unit(value) for value in METRIC_UNITS.get(metric, [])]
                    if allowed_units and normalized_unit not in allowed_units:
                        continue
                    for entry in entries:
                        if entry.get('form') != form or not isinstance(entry.get('val'), (int, float)):
                            continue
                        output.append(
                            FinancialFact(
                                metric=metric,
                                value=float(entry['val']),
                                unit=normalized_unit,
                                fiscal_year=self._normalized_fiscal_year(
                                    entry,
                                    form,
                                ),
                                fiscal_period=entry.get('fp'),
                                form=form,
                                filed=entry.get('filed', ''),
                                start=entry.get('start'),
                                end=entry.get('end'),
                                accession_number=entry.get('accn'),
                                concept=concept,
                                taxonomy=taxonomy,
                            )
                        )
        return output

    def _prefer_concepts(
        self,
        facts: list[FinancialFact],
        metric: str,
        form: str,
    ) -> list[FinancialFact]:
        """Keep the highest-priority concept for each reporting period."""

        concept_rank = {
            concept: rank
            for rank, concept in enumerate(METRIC_CONCEPTS[metric])
        }
        taxonomy_rank = {
            taxonomy: rank
            for rank, taxonomy in enumerate(self.TAXONOMIES)
        }

        by_period: dict[tuple, list[FinancialFact]] = {}
        without_period: list[FinancialFact] = []

        for fact in facts:
            if fact.fiscal_year is None:
                without_period.append(fact)
                continue

            period_key = (
                (fact.fiscal_year, fact.fiscal_period, fact.end)
                if form == '10-Q'
                else (fact.fiscal_year,)
            )

            by_period.setdefault(period_key, []).append(fact)

        selected: list[FinancialFact] = []

        for _, candidates in by_period.items():
            best_rank = min(
                concept_rank.get(fact.concept, 999)
                for fact in candidates
            )
            candidates = [
                fact
                for fact in candidates
                if concept_rank.get(fact.concept, 999) == best_rank
            ]

            best_taxonomy = min(
                taxonomy_rank.get(fact.taxonomy, 999)
                for fact in candidates
            )
            candidates = [
                fact
                for fact in candidates
                if taxonomy_rank.get(fact.taxonomy, 999)
                == best_taxonomy
            ]

            candidates.sort(
                key=lambda fact: (
                    fact.filed,
                    fact.end or '',
                    fact.start or '',
                ),
                reverse=True,
            )

            selected.append(candidates[0])

        if not selected and without_period:
            without_period.sort(
                key=lambda fact: (
                    -concept_rank.get(fact.concept, 999),
                    fact.filed,
                    fact.end or '',
                ),
                reverse=True,
            )
            selected.append(without_period[0])

        return selected

    def get_metric(self, cik: str, metric: str, fiscal_year: int | None = None, form: str = '10-K') -> FinancialFact:
        if metric not in METRIC_CONCEPTS:
            raise ValueError(f'Unsupported metric: {metric}')
        facts = self.client.get_company_facts(cik)
        candidates = self._prefer_period(self._iter_metric_facts(facts, metric, form), metric, form)
        candidates = self._prefer_concepts(candidates, metric, form)
        if fiscal_year is not None:
            candidates = [fact for fact in candidates if fact.fiscal_year == fiscal_year]
        if not candidates:
            raise FinancialFactNotFound(f'No SEC XBRL fact found for {metric} ({form}, FY={fiscal_year or "latest"})')
        candidates.sort(key=lambda fact: (fact.fiscal_year or 0, fact.end or '', fact.filed), reverse=True)
        return candidates[0]

    def get_series(self, cik: str, metric: str, form: str = '10-K', limit: int = 5) -> list[FinancialFact]:
        if metric not in METRIC_CONCEPTS:
            raise ValueError(f'Unsupported metric: {metric}')
        facts = self.client.get_company_facts(cik)
        all_facts = self._prefer_period(self._iter_metric_facts(facts, metric, form), metric, form)
        all_facts = self._prefer_concepts(all_facts, metric, form)
        return sorted(all_facts, key=lambda fact: (fact.fiscal_year or 0, fact.end or ''), reverse=True)[:limit]

    @staticmethod
    def _prefer_period(
        facts: list[FinancialFact],
        metric: str,
        form: str,
    ) -> list[FinancialFact]:

        duration_metrics = {
            'revenue',
            'net_income',
            'operating_income',
            'gross_profit',
            'operating_cash_flow',
            'eps',
        }

        if form == '10-K':
            fy_facts = [
                fact
                for fact in facts
                if fact.fiscal_period == 'FY'
            ]

            if fy_facts:
                facts = fy_facts

            if metric in duration_metrics:
                annual: list[FinancialFact] = []

                for fact in facts:
                    if not fact.start or not fact.end:
                        continue

                    try:
                        days = (
                            date.fromisoformat(fact.end)
                            - date.fromisoformat(fact.start)
                        ).days
                    except ValueError:
                        continue

                    if 300 <= days <= 380:
                        annual.append(fact)

                if annual:
                    facts = annual

            return facts

        if form != '10-Q':
            return facts

        # -------------------------------------------------
        # 10-Q: first remove comparative contexts.
        #
        # One 10-Q accession commonly contains:
        #   - the current quarter/current balance, and
        #   - prior-year comparative values.
        #
        # Within the same accession/concept/unit/fiscal
        # period, the fact with the latest period end is
        # the current reporting context.
        # -------------------------------------------------

        grouped: dict[tuple, list[FinancialFact]] = {}

        for fact in facts:
            key = (
                fact.accession_number,
                fact.fiscal_period,
                fact.concept,
                fact.taxonomy,
                fact.unit,
            )
            grouped.setdefault(key, []).append(fact)

        primary_contexts: list[FinancialFact] = []

        for candidates in grouped.values():
            valid_ends = [
                fact.end
                for fact in candidates
                if fact.end
            ]

            if not valid_ends:
                primary_contexts.extend(candidates)
                continue

            latest_end = max(valid_ends)

            primary_contexts.extend(
                fact
                for fact in candidates
                if fact.end == latest_end
            )

        facts = primary_contexts

        if metric not in duration_metrics:
            return facts

        # -------------------------------------------------
        # Duration metrics:
        #
        # Revenue / income / EPS commonly provide both a
        # discrete quarter and a YTD context in Q2/Q3.
        # Prefer the discrete ~3-month context.
        #
        # Cash-flow statements commonly provide only YTD
        # Q2/Q3 values. If no discrete quarter exists, keep
        # the shortest available current context rather than
        # dropping the period completely.
        # -------------------------------------------------

        period_groups: dict[tuple, list[tuple[FinancialFact, int]]] = {}

        for fact in facts:
            if not fact.start or not fact.end:
                continue

            try:
                days = (
                    date.fromisoformat(fact.end)
                    - date.fromisoformat(fact.start)
                ).days
            except ValueError:
                continue

            key = (
                fact.accession_number,
                fact.fiscal_period,
                fact.concept,
                fact.taxonomy,
                fact.unit,
                fact.end,
            )

            period_groups.setdefault(key, []).append(
                (fact, days)
            )

        selected: list[FinancialFact] = []

        for candidates in period_groups.values():
            discrete = [
                (fact, days)
                for fact, days in candidates
                if 60 <= days <= 120
            ]

            if discrete:
                selected.extend(
                    fact
                    for fact, _ in discrete
                )
                continue

            # No discrete quarter exists, as is common for
            # operating cash flow in Q2/Q3. Preserve the
            # shortest current YTD context.
            candidates.sort(
                key=lambda item: item[1]
            )
            selected.append(candidates[0][0])

        return selected or facts
