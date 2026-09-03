from __future__ import annotations


def yoy_growth(current: float, previous: float) -> float:
    if previous == 0:
        raise ZeroDivisionError('Cannot compute growth from a zero previous value')
    return (current - previous) / previous * 100.0


def margin(profit: float, revenue: float) -> float:
    if revenue == 0:
        raise ZeroDivisionError('Cannot compute margin with zero revenue')
    return profit / revenue * 100.0


def difference(current: float, previous: float) -> float:
    return current - previous
