"""Execution tracing.

An agent system you cannot inspect is not much better than a prompt in a shell
script. Every step the orchestrator takes lands here as a :class:`TraceEvent`,
which is rendered to Markdown for humans and to JSON for machines.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    offset_ms: float
    kind: str
    actor: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_ms": round(self.offset_ms, 1),
            "kind": self.kind,
            "actor": self.actor,
            "summary": self.summary,
            "payload": self.payload,
        }


class Trace:
    """Ordered event log with relative timestamps."""

    def __init__(self, title: str = "run") -> None:
        self.title = title
        self._started = time.perf_counter()
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def record(self, kind: str, actor: str, summary: str, **payload: Any) -> TraceEvent:
        event = TraceEvent(
            offset_ms=(time.perf_counter() - self._started) * 1000,
            kind=kind,
            actor=actor,
            summary=summary,
            payload={key: value for key, value in payload.items() if value is not None},
        )
        self._events.append(event)
        return event

    def span(self, kind: str, actor: str, summary: str) -> float:
        """Record a start marker and return the clock for :meth:`close_span`."""
        self.record(f"{kind}.start", actor, summary)
        return time.perf_counter()

    def close_span(
        self, started: float, kind: str, actor: str, summary: str, **payload: Any
    ) -> None:
        self.record(
            f"{kind}.end",
            actor,
            summary,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            **payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "events": [event.to_dict() for event in self._events]}

    def to_markdown(self) -> str:
        rows = [
            "| # | offset (ms) | event | actor | summary |",
            "|---|---|---|---|---|",
        ]
        for index, event in enumerate(self._events, start=1):
            summary = event.summary.replace("|", "\\|")
            rows.append(
                f"| {index} | {event.offset_ms:.0f} | `{event.kind}` | {event.actor} | {summary} |"
            )
        return "\n".join(rows)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target
