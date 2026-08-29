from __future__ import annotations

from typing import Any


def compare_tool_sequence(observed: list[str], expected: list[str]) -> dict[str, Any]:
    """Compare an observed tool sequence with a reference sequence."""
    matches = sum(a == b for a, b in zip(observed, expected))
    return {
        "observed": observed,
        "expected": expected,
        "matching_positions": matches,
        "expected_steps": len(expected),
        "sequence_accuracy": matches / len(expected) if expected else 1.0,
        "exact_match": observed == expected,
    }


def evaluate_agent_result(
    *,
    observed_tools: list[str],
    expected_tools: list[str],
    conclusion: str | None,
) -> dict[str, Any]:
    sequence = compare_tool_sequence(observed_tools, expected_tools)
    return {
        "tool_sequence": sequence,
        "has_conclusion": bool(conclusion and conclusion.strip()),
    }
