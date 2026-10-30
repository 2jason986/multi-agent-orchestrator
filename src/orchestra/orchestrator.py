"""The orchestration loop.

Four roles, one conversation, a bounded revision loop:

    planner  -> decides priority and search queries
    researcher -> calls the knowledge-base tool until it has evidence
    responder -> writes the customer reply from that evidence
    reviewer -> approves, or sends it back with feedback (max N rounds)

The orchestrator owns the data flow and the trace; agents never talk to each
other directly, which keeps the control flow readable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import Agent, build_agents
from .builtin_tools import kb_search_tool, ticket_lookup_tool
from .knowledge import KnowledgeBase
from .llm import LLMClient
from .tickets import Ticket, TicketStore
from .tools import ToolRegistry
from .trace import Trace


@dataclass(slots=True)
class RunResult:
    ticket_id: str
    subject: str
    priority: str
    reply: str
    approved: bool
    rounds: int
    citations: list[str] = field(default_factory=list)
    feedback: str = ""
    trace: Trace | None = None

    def to_markdown(self) -> str:
        status = "approved" if self.approved else "sent for human review"
        citations = ", ".join(f"`{item}`" for item in self.citations) or "none"
        parts = [
            f"## {self.ticket_id} · {self.subject}",
            "",
            f"- priority: **{self.priority}**",
            f"- review rounds: {self.rounds} ({status})",
            f"- citations: {citations}",
            "",
            "### Reply",
            "",
            self.reply.strip(),
            "",
            "### Trace",
            "",
            self.trace.to_markdown() if self.trace else "_no trace_",
            "",
        ]
        return "\n".join(parts)


class Orchestrator:
    """Wires agents, tools and a tracer into a single callable pipeline."""

    def __init__(
        self,
        llm: LLMClient,
        knowledge_base: KnowledgeBase,
        ticket_store: TicketStore,
        *,
        max_review_rounds: int = 2,
        agents: dict[str, Agent] | None = None,
    ) -> None:
        registry = ToolRegistry()
        registry.register(kb_search_tool(knowledge_base))
        registry.register(ticket_lookup_tool(ticket_store))
        self.registry = registry
        self.ticket_store = ticket_store
        self.max_review_rounds = max_review_rounds
        everything = agents or build_agents(llm, registry)
        self.planner = everything["planner"]
        self.researcher = everything["researcher"]
        self.responder = everything["responder"]
        self.reviewer = everything["reviewer"]

    def handle(self, ticket: Ticket) -> RunResult:
        trace = Trace(title=f"{ticket.id} · {ticket.subject}")
        trace.record("ticket.received", "orchestrator", ticket.subject, ticket_id=ticket.id)
        prompt = ticket.to_prompt()

        started = trace.span("agent", "planner", "classify and break down the request")
        plan = self.planner.run(prompt)
        priority = str(plan.get("priority", "normal"))
        queries = [str(item) for item in plan.get("queries") or [ticket.subject]]
        trace.close_span(
            started,
            "agent",
            "planner",
            f"priority={priority}",
            queries=queries,
            plan=plan.get("plan"),
        )

        started = trace.span("agent", "researcher", f"search knowledge base for {queries[0]!r}")
        research = self.researcher.run(prompt, context={"QUERY": queries[0]})
        for result in research.tool_results:
            trace.record(
                "tool.call",
                result.name,
                f"{'ok' if result.ok else 'failed'}: {result.arguments.get('query', '')}",
                arguments=result.arguments,
                error=result.error,
            )
        notes = str(research.get("summary", ""))
        citations = [str(item) for item in research.get("citations") or []]
        trace.close_span(
            started,
            "agent",
            "researcher",
            f"{len(research.tool_results)} tool call(s), {len(citations)} citation(s)",
        )

        started = trace.span("agent", "responder", "draft the customer reply")
        draft = self.responder.run(prompt, context={"RESEARCH_NOTES": notes})
        reply = str(draft.get("draft", ""))
        citations = [str(item) for item in draft.get("citations") or citations]
        trace.close_span(started, "agent", "responder", f"{len(reply)} characters drafted")

        rounds = 0
        approved = False
        feedback = ""
        while rounds < self.max_review_rounds:
            rounds += 1
            started = trace.span("agent", "reviewer", f"review round {rounds}")
            review = self.reviewer.run(
                prompt, context={"DRAFT": reply, "REVIEW_ROUND": rounds}
            )
            verdict = str(review.get("verdict", "revise"))
            feedback = str(review.get("feedback", ""))
            trace.close_span(started, "agent", "reviewer", f"verdict={verdict}", feedback=feedback)
            if verdict == "approve":
                approved = True
                break
            started = trace.span("agent", "responder", f"revise after round {rounds} review")
            revision = self.responder.run(
                prompt, context={"RESEARCH_NOTES": notes, "REVIEW_FEEDBACK": feedback}
            )
            reply = str(revision.get("draft") or reply)
            citations = [str(item) for item in revision.get("citations") or citations]
            trace.close_span(started, "agent", "responder", "reply revised")

        self.ticket_store.set_status(ticket.id, "answered" if approved else "needs_human")
        trace.record(
            "run.finished",
            "orchestrator",
            f"approved={approved} rounds={rounds}",
            ticket_id=ticket.id,
        )
        return RunResult(
            ticket_id=ticket.id,
            subject=ticket.subject,
            priority=priority,
            reply=reply,
            approved=approved,
            rounds=rounds,
            citations=citations,
            feedback=feedback,
            trace=trace,
        )

    def handle_all(self, status: str = "open") -> list[RunResult]:
        tickets = self.ticket_store.open_tickets() if status == "open" else self.ticket_store.all()
        return [self.handle(ticket) for ticket in tickets]


def summarise(results: list[RunResult]) -> dict[str, Any]:
    """Small aggregate used by the CLI report and the tests."""
    return {
        "tickets": len(results),
        "approved": sum(1 for result in results if result.approved),
        "needs_human": sum(1 for result in results if not result.approved),
        "rounds": sum(result.rounds for result in results),
    }
