import pandas as pd

from sciagent.execution import ExecutionTrace, ScientificExecutor
from sciagent.tools.default_registry import build_default_registry


def test_execution_records_tool_calls():
    registry = build_default_registry()
    trace = ExecutionTrace()
    executor = ScientificExecutor(registry, trace)

    series = pd.Series([1.0, 2.0, 3.0])
    result = executor.execute("summary_statistics", series=series)

    assert result["mean"] == 2.0
    assert len(trace.calls) == 1
    assert trace.calls[0].step == 1
    assert trace.calls[0].tool == "summary_statistics"
    assert trace.calls[0].arguments["series"].tolist() == [1.0, 2.0, 3.0]
