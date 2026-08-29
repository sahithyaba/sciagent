import pandas as pd

from sciagent.agent import ScientificAgent
from sciagent.tools.default_registry import build_default_registry


class MockBackend:
    def __init__(self):
        self.calls = 0

    def next_action(self, question, observations, tools):
        self.calls += 1
        if self.calls == 1:
            return {"type": "tool", "name": "summary_statistics", "arguments": {"series": pd.Series([1.0, 2.0, 3.0])}}
        return {"type": "final", "conclusion": "The mean is 2.0."}


def test_agent_runs_tool_then_concludes():
    agent = ScientificAgent(MockBackend(), build_default_registry(), max_steps=3)
    result = agent.run("What is the mean?", {"columns": ["value"]})

    assert result.conclusion == "The mean is 2.0."
    assert len(result.trace["tool_calls"]) == 1
    assert result.trace["tool_calls"][0]["tool"] == "summary_statistics"


def test_agent_stops_at_max_steps():
    class NeverEndingBackend:
        def next_action(self, question, observations, tools):
            return {"type": "tool", "name": "summary_statistics", "arguments": {"series": pd.Series([1.0])}}

    agent = ScientificAgent(NeverEndingBackend(), build_default_registry(), max_steps=2)
    try:
        agent.run("Keep going", {})
    except RuntimeError as exc:
        assert "max_steps=2" in str(exc)
    else:
        raise AssertionError("Expected max-step RuntimeError")
