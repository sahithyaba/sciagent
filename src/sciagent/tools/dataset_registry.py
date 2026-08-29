from __future__ import annotations

import pandas as pd

from .data import inspect_dataset
from .numerical import detect_zscore_anomalies
from .registry import Tool, ToolRegistry
from .statistics import pearson_correlation, summary_statistics, welch_t_test


def build_dataset_registry(df: pd.DataFrame) -> ToolRegistry:
    """Build a safe registry whose tools address columns in one loaded dataset.

    The model supplies column names rather than Python objects or filesystem
    paths. This keeps execution deterministic and prevents the LLM from choosing
    arbitrary files on the host machine.
    """
    registry = ToolRegistry()

    registry.register(Tool(
        "inspect_dataset",
        "Inspect the loaded dataset schema, size, dtypes, and missing values. Takes no arguments.",
        lambda: inspect_dataset(df),
    ))
    registry.register(Tool(
        "summary_statistics",
        "Calculate descriptive statistics. Arguments: column (string column name).",
        lambda column: summary_statistics(df[column]),
    ))
    registry.register(Tool(
        "detect_zscore_anomalies",
        "Detect observations above an absolute z-score threshold. Arguments: column (string), threshold (number, default 3.0).",
        lambda column, threshold=3.0: detect_zscore_anomalies(df[column], threshold),
    ))
    registry.register(Tool(
        "welch_t_test",
        "Compare two independent samples. Arguments: a (string column), b (string column).",
        lambda a, b: welch_t_test(df[a], df[b]),
    ))
    registry.register(Tool(
        "pearson_correlation",
        "Calculate Pearson correlation and p-value. Arguments: x (string column), y (string column).",
        lambda x, y: pearson_correlation(df[x], df[y]),
    ))
    return registry
