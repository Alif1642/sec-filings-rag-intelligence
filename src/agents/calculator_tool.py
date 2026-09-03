from __future__ import annotations

from src.financials.calculations import difference, margin, yoy_growth


class FinancialCalculatorTool:
    OPERATIONS = {
        'yoy_growth': yoy_growth,
        'margin': margin,
        'difference': difference,
    }

    def calculate_metric(self, operation: str, a: float, b: float) -> dict:
        if operation not in self.OPERATIONS:
            raise ValueError(f'Unsupported financial calculation: {operation}')
        result = self.OPERATIONS[operation](a, b)
        return {'operation': operation, 'inputs': [a, b], 'value': result, 'source': 'Deterministic Python calculation'}
