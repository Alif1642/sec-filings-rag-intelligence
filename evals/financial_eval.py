from __future__ import annotations


def financial_errors(predicted: float, ground_truth: float) -> dict[str, float | bool]:
    absolute = abs(predicted - ground_truth)
    percentage = absolute / abs(ground_truth) * 100.0 if ground_truth != 0 else (0.0 if absolute == 0 else float('inf'))
    return {
        'exact_match': predicted == ground_truth,
        'absolute_error': absolute,
        'percentage_error': percentage,
    }
