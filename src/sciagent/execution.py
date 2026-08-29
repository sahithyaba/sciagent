from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import json

from .tools.registry import ToolRegistry


@dataclass
class ToolCall:
    step: int
    tool: str
    arguments: dict[str, Any]
    result: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExecutionTrace:
    """Append-only record of an agent's scientific tool interactions."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def record(self, tool: str, arguments: dict[str, Any], result: Any) -> None:
        self.calls.append(
            ToolCall(
                step=len(self.calls) + 1,
                tool=tool,
                arguments=arguments,
                result=result,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"tool_calls": [asdict(call) for call in self.calls]}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ScientificExecutor:
    """Execute only tools registered in the scientific tool registry."""

    def __init__(self, registry: ToolRegistry, trace: ExecutionTrace | None = None):
        self.registry = registry
        self.trace = trace or ExecutionTrace()

    def execute(self, tool_name: str, **arguments: Any) -> Any:
        result = self.registry.execute(tool_name, **arguments)
        self.trace.record(tool_name, arguments, result)
        return result
