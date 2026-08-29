import pandas as pd
import pytest

from sciagent.tools.default_registry import build_default_registry


def test_default_registry_exposes_scientific_tools():
    registry = build_default_registry()
    names = {tool["name"] for tool in registry.list_tools()}
    assert {
        "load_dataset",
        "inspect_dataset",
        "summary_statistics",
        "detect_zscore_anomalies",
        "welch_t_test",
        "pearson_correlation",
    } <= names


def test_unknown_tool_is_rejected():
    registry = build_default_registry()
    with pytest.raises(KeyError):
        registry.execute("not_a_scientific_tool")


def test_tool_execution_is_deterministic_for_same_input():
    registry = build_default_registry()
    series = pd.Series([1.0, 1.0, 1.0])
    result = registry.execute("summary_statistics", series=series)
    assert result["n"] == 3
    assert result["mean"] == 1.0
