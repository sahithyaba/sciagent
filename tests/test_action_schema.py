import pytest

from sciagent.backends.action_schema import validate_action


TOOLS = {"summary_statistics", "detect_zscore_anomalies"}


def test_valid_tool_action():
    action = validate_action(
        {"type": "tool", "name": "summary_statistics", "arguments": {"x": 1}},
        TOOLS,
    )
    assert action["type"] == "tool"
    assert action["name"] == "summary_statistics"


def test_unknown_tool_is_rejected_before_execution():
    with pytest.raises(ValueError, match="Unknown scientific tool"):
        validate_action({"type": "tool", "name": "shell", "arguments": {}}, TOOLS)


def test_final_action_requires_conclusion():
    with pytest.raises(ValueError, match="non-empty conclusion"):
        validate_action({"type": "final", "conclusion": ""}, TOOLS)
