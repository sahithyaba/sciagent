from .data import inspect_dataset, load_dataset
from .numerical import detect_zscore_anomalies
from .statistics import pearson_correlation, summary_statistics, welch_t_test
from .registry import Tool, ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("load_dataset", "Load a CSV scientific dataset.", load_dataset))
    registry.register(Tool("inspect_dataset", "Inspect schema, size, dtypes, and missing values.", inspect_dataset))
    registry.register(Tool("summary_statistics", "Calculate descriptive statistics for a numeric series.", summary_statistics))
    registry.register(Tool("detect_zscore_anomalies", "Detect observations above an absolute z-score threshold.", detect_zscore_anomalies))
    registry.register(Tool("welch_t_test", "Compare two independent samples using Welch's t-test.", welch_t_test))
    registry.register(Tool("pearson_correlation", "Calculate Pearson correlation and p-value.", pearson_correlation))
    return registry
