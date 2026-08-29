from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .metrics import evaluate_agent_result


def load_case(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_case(
    case: dict[str, Any],
    *,
    observed_tools: list[str],
    conclusion: str | None,
) -> dict[str, Any]:
    """Evaluate the observable parts of one benchmark case.

    Numerical and scientific-conclusion scoring will be added after the first
    model-backed runs, when we can define robust tolerances and semantic checks.
    """
    expected = case["reference_analysis"]
    # Keep the reference order explicit and reproducible.
    expected_tools = [
        "detect_zscore_anomalies",
        "welch_t_test",
        "pearson_correlation",
    ]
    result = evaluate_agent_result(
        observed_tools=observed_tools,
        expected_tools=expected_tools,
        conclusion=conclusion,
    )
    result["benchmark_id"] = case["id"]
    result["reference_methods"] = expected
    return result
