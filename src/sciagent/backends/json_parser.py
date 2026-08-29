from __future__ import annotations

import json
from typing import Any


def parse_json_action(response: str) -> dict[str, Any]:
    """Parse a model response containing one JSON agent action."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        action = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc

    if not isinstance(action, dict):
        raise ValueError("LLM JSON response must be an object")
    return action
