from __future__ import annotations

METRIC_CONCEPTS: dict[str, list[str]] = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'],
    'net_income': ['NetIncomeLoss', 'ProfitLoss'],
    'operating_income': ['OperatingIncomeLoss'],
    'gross_profit': ['GrossProfit'],
    'assets': ['Assets'],
    'liabilities': ['Liabilities'],
    'cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities'],
    'eps': ['EarningsPerShareDiluted', 'EarningsPerShareBasic'],
    'shares_outstanding': ['CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding'],
}

METRIC_UNITS: dict[str, list[str]] = {
    'eps': ['USD/shares', 'USD / shares'],
    'shares_outstanding': ['shares'],
}
