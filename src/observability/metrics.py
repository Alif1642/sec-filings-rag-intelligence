from __future__ import annotations

from src.config import Settings, get_settings


def estimate_cost(input_tokens: int, output_tokens: int, settings: Settings | None = None) -> float | None:
    settings = settings or get_settings()
    if settings.pricing_input_per_1m is None or settings.pricing_output_per_1m is None:
        return None
    return (input_tokens / 1_000_000) * settings.pricing_input_per_1m + (output_tokens / 1_000_000) * settings.pricing_output_per_1m
