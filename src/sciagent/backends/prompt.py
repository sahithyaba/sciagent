from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a scientific computing agent.

Solve the user's scientific question by selecting tools from the supplied
scientific tool registry. Do not invent tool names and do not execute arbitrary
code. Prefer evidence from tool results over unsupported claims.

Return exactly one JSON object with one of these shapes:
{"type":"tool","name":"TOOL_NAME","arguments":{...}}
{"type":"final","conclusion":"..."}

Use a tool action when more computation or evidence is needed. Use final only
when you have enough evidence to answer the scientific question. State
statistical conclusions carefully and distinguish correlation from causation.
"""


def build_prompt(
    question: str,
    observations: list[dict[str, Any]],
    tools: list[dict[str, str]],
) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\nAvailable tools:\n"
        + json.dumps(tools, indent=2, default=str)
        + "\n\nScientific question:\n"
        + question
        + "\n\nObservations so far:\n"
        + json.dumps(observations, indent=2, default=str)
        + "\n\nReturn only the JSON action."
    )
