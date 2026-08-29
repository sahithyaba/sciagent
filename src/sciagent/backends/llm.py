from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class LLMBackend:
    """Model-agnostic adapter for a callable that returns a structured action.

    The callable receives the scientific question, observations, and available
    tools. A concrete provider adapter can wrap an OpenAI-compatible endpoint,
    a local model, or another inference service without changing ScientificAgent.
    """

    invoke: Callable[[str, list[dict[str, Any]], list[dict[str, str]]], dict[str, Any]]

    def next_action(
        self,
        question: str,
        observations: list[dict[str, Any]],
        tools: list[dict[str, str]],
    ) -> dict[str, Any]:
        action = self.invoke(question, observations, tools)
        if not isinstance(action, dict):
            raise ValueError("LLM backend must return a dictionary action")
        return action
