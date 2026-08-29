from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .execution import ScientificExecutor
from .tools.registry import ToolRegistry


class AgentBackend(Protocol):
    """Protocol implemented by an LLM or deterministic planning backend."""

    def next_action(
        self,
        question: str,
        observations: list[dict[str, Any]],
        tools: list[dict[str, str]],
    ) -> dict[str, Any]: ...


@dataclass
class AgentResult:
    conclusion: str | None
    observations: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


class ScientificAgent:
    """Model-agnostic agent loop for controlled scientific tool use."""

    def __init__(self, backend: AgentBackend, registry: ToolRegistry, *, max_steps: int = 10) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.backend = backend
        self.registry = registry
        self.executor = ScientificExecutor(registry)
        self.max_steps = max_steps

    def run(self, question: str, dataset_description: dict[str, Any]) -> AgentResult:
        observations: list[dict[str, Any]] = [{"type": "dataset", "description": dataset_description}]

        for _ in range(self.max_steps):
            action = self.backend.next_action(
                question=question,
                observations=observations,
                tools=self.registry.list_tools(),
            )
            action_type = action.get("type")

            if action_type == "final":
                return AgentResult(
                    conclusion=str(action.get("conclusion", "")),
                    observations=observations,
                    trace=self.executor.trace.to_dict(),
                )

            if action_type != "tool":
                raise ValueError(f"Unsupported action type: {action_type!r}")

            name = action.get("name")
            arguments = action.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("Tool action requires a name and dictionary arguments")

            result = self.executor.execute(name, **arguments)
            observations.append({
                "type": "tool_result",
                "tool": name,
                "arguments": arguments,
                "result": result,
            })

        raise RuntimeError(f"Agent reached max_steps={self.max_steps} without producing a conclusion")
