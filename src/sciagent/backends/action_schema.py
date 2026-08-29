from __future__ import annotations

from typing import Any


ALLOWED_ACTION_TYPES = {"tool", "final"}


def validate_action(action: dict[str, Any], available_tools: set[str]) -> dict[str, Any]:
    """Validate and normalize one model-produced agent action."""
    if not isinstance(action, dict):
        raise ValueError("Agent action must be a dictionary")

    action_type = action.get("type")
    if action_type not in ALLOWED_ACTION_TYPES:
        raise ValueError(f"Invalid action type: {action_type!r}")

    if action_type == "final":
        conclusion = action.get("conclusion")
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise ValueError("Final action requires a non-empty conclusion")
        return {"type": "final", "conclusion": conclusion.strip()}

    name = action.get("name")
    arguments = action.get("arguments", {})
    if not isinstance(name, str) or name not in available_tools:
        raise ValueError(f"Unknown scientific tool: {name!r}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary")

    return {"type": "tool", "name": name, "arguments": arguments}
