from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """A scientific operation exposed to an agent."""

    name: str
    description: str
    function: Callable[..., Any]


class ToolRegistry:
    """Controlled registry of scientific tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown scientific tool: {name}") from exc

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def execute(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).function(**kwargs)
