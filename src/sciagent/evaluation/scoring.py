from __future__ import annotations

from typing import Any


def within_tolerance(actual: float, expected: float, *, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    return abs(actual - expected) <= max(atol, rtol * abs(expected))


def score_numeric(actual: float, expected: float, *, rtol: float = 1e-6, atol: float = 1e-9) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected": expected,
        "within_tolerance": within_tolerance(actual, expected, rtol=rtol, atol=atol),
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
    }


def score_anomaly_indices(actual: list[int], expected: list[int]) -> dict[str, Any]:
    actual_set, expected_set = set(actual), set(expected)
    intersection = actual_set & expected_set
    precision = len(intersection) / len(actual_set) if actual_set else 1.0 if not expected_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "actual": sorted(actual_set),
        "expected": sorted(expected_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": actual_set == expected_set,
    }
