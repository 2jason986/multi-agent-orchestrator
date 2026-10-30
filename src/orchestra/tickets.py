"""Ticket domain model and an in-memory store.

Deliberately boring: the point of this repository is the orchestration layer, so
the data layer stays a JSON file plus a handful of dict lookups.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Ticket:
    id: str
    subject: str
    body: str
    customer: str = ""
    channel: str = "email"
    created_at: str = ""
    status: str = "open"
    tags: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [
            f"TICKET_ID: {self.id}",
            f"CHANNEL: {self.channel}",
            f"SUBJECT: {self.subject}",
            "BODY:",
            self.body.strip(),
        ]
        if self.customer:
            lines.insert(2, f"CUSTOMER: {self.customer}")
        if self.tags:
            lines.append("TAGS: " + ", ".join(self.tags))
        return "\n".join(lines)


class TicketStore:
    def __init__(self, tickets: list[Ticket]) -> None:
        self._tickets = {ticket.id: ticket for ticket in tickets}

    @classmethod
    def from_json(cls, path: str | Path) -> TicketStore:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tickets = [Ticket(**item) for item in raw["tickets"]]
        return cls(tickets)

    def all(self) -> list[Ticket]:
        return list(self._tickets.values())

    def open_tickets(self) -> list[Ticket]:
        return [ticket for ticket in self._tickets.values() if ticket.status == "open"]

    def get(self, identifier: str) -> Ticket:
        try:
            return self._tickets[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown ticket {identifier!r}") from exc

    def set_status(self, identifier: str, status: str) -> None:
        self.get(identifier).status = status

    def to_dict(self) -> dict[str, Any]:
        return {"tickets": [asdict(ticket) for ticket in self._tickets.values()]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
