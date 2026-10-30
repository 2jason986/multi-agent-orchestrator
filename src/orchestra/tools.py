"""Tool layer: the only way an agent can touch the outside world.

Keeping tools behind a registry gives three things for free: a single place to
validate arguments, a stable schema to describe the tool to the model, and a
recorded ``ToolResult`` for every call so runs stay auditable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Schema handed to the model.

    ``parameters`` maps an argument name to a short human description; the
    orchestrator only needs it for prompt construction and validation.
    """

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    output: Any
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.ok:
            payload["results"] = self.output if isinstance(self.output, list) else [self.output]
        else:
            payload["error"] = self.error
        return payload


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    def __call__(self, **kwargs: Any) -> Any: ...


class ToolRegistry:
    """Name -> callable lookup with uniform error handling."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Tool {tool.spec.name!r} is already registered")
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool {name!r}. Available: {sorted(self._tools)}") from exc

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def describe(self) -> str:
        lines = []
        for spec in self.specs():
            args = ", ".join(spec.parameters) or "no arguments"
            lines.append(f"- {spec.name}({args}): {spec.description}")
        return "\n".join(lines)

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult(name=name, arguments=arguments, output=None, ok=False, error=str(exc))
        spec = tool.spec
        if spec.parameters:
            # Models love to invent extra fields; keep only what the tool declares.
            arguments = {key: value for key, value in arguments.items() if key in spec.parameters}
        try:
            return ToolResult(name=name, arguments=arguments, output=tool(**arguments))
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return ToolResult(
                name=name,
                arguments=arguments,
                output=None,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )


@dataclass(slots=True)
class FunctionTool:
    """Adapter turning a plain function into a registry entry."""

    spec: ToolSpec
    func: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)
