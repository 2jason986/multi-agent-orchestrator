"""Tools the demo agents are allowed to use."""

from __future__ import annotations

from typing import Any

from .knowledge import KnowledgeBase
from .tickets import TicketStore
from .tools import FunctionTool, ToolSpec


def kb_search_tool(knowledge_base: KnowledgeBase) -> FunctionTool:
    """Expose the retriever as a tool the model can decide to call."""

    def _search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
        limit = max(1, min(int(top_k), 10))
        return [hit.to_dict() for hit in knowledge_base.search(query, top_k=limit)]

    return FunctionTool(
        spec=ToolSpec(
            name="kb_search",
            description=(
                "Search the support knowledge base and return matching notes "
                "with a grounded excerpt and a similarity score."
            ),
            parameters={
                "query": "natural-language search string",
                "top_k": "how many notes to return, 1-10 (default 3)",
            },
        ),
        func=_search,
    )


def ticket_lookup_tool(store: TicketStore) -> FunctionTool:
    """Read-only view over the ticket store, useful when an agent needs context."""

    def _lookup(ticket_id: str) -> dict[str, Any]:
        ticket = store.get(ticket_id)
        return {
            "id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status,
            "channel": ticket.channel,
            "tags": ticket.tags,
        }

    return FunctionTool(
        spec=ToolSpec(
            name="ticket_lookup",
            description="Look up a ticket record by id.",
            parameters={"ticket_id": "ticket identifier such as T-1001"},
        ),
        func=_lookup,
    )
