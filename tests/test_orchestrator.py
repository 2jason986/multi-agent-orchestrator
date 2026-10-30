"""End-to-end tests for the orchestration loop, driven by the mock model."""

from __future__ import annotations

import unittest
from pathlib import Path

from orchestra.knowledge import KnowledgeBase
from orchestra.mock_llm import MockClient
from orchestra.orchestrator import Orchestrator, summarise
from orchestra.tickets import TicketStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def build_orchestrator(**kwargs: object) -> Orchestrator:
    return Orchestrator(
        llm=MockClient(),
        knowledge_base=KnowledgeBase.from_directory(DATA / "knowledge_base"),
        ticket_store=TicketStore.from_json(DATA / "tickets.json"),
        **kwargs,  # type: ignore[arg-type]
    )


class OrchestratorTests(unittest.TestCase):
    def test_escalates_priority_for_a_production_outage(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.handle(orchestrator.ticket_store.get("T-1002"))
        self.assertEqual(result.priority, "high")

    def test_revision_loop_ends_in_approval(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.handle(orchestrator.ticket_store.get("T-1001"))
        self.assertTrue(result.approved)
        self.assertEqual(result.rounds, 2)
        self.assertTrue(result.reply.strip())

    def test_citations_come_from_the_knowledge_base(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.handle(orchestrator.ticket_store.get("T-1001"))
        self.assertIn("kb-export", result.citations)
        self.assertIn("[kb-export]", result.reply)

    def test_ticket_status_is_updated(self) -> None:
        orchestrator = build_orchestrator()
        ticket = orchestrator.ticket_store.get("T-1003")
        orchestrator.handle(ticket)
        self.assertEqual(ticket.status, "answered")

    def test_trace_starts_and_ends_in_order(self) -> None:
        orchestrator = build_orchestrator()
        result = orchestrator.handle(orchestrator.ticket_store.get("T-1003"))
        events = result.trace.events if result.trace else []
        self.assertEqual(events[0].kind, "ticket.received")
        self.assertEqual(events[-1].kind, "run.finished")
        kinds = [event.kind for event in events]
        self.assertIn("tool.call", kinds)
        self.assertEqual(kinds.count("agent.start"), kinds.count("agent.end"))

    def test_budget_exhaustion_hands_the_ticket_to_a_human(self) -> None:
        orchestrator = build_orchestrator(max_review_rounds=1)
        ticket = orchestrator.ticket_store.get("T-1002")
        result = orchestrator.handle(ticket)
        self.assertFalse(result.approved)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(ticket.status, "needs_human")

    def test_handle_all_covers_every_open_ticket(self) -> None:
        orchestrator = build_orchestrator()
        results = orchestrator.handle_all()
        self.assertEqual(len(results), 3)
        summary = summarise(results)
        self.assertEqual(summary["tickets"], 3)
        self.assertEqual(summary["approved"], 3)
        self.assertEqual(summary["rounds"], 6)

    def test_markdown_report_contains_trace_table(self) -> None:
        orchestrator = build_orchestrator()
        markdown = orchestrator.handle(orchestrator.ticket_store.get("T-1001")).to_markdown()
        self.assertIn("### Reply", markdown)
        self.assertIn("| # | offset (ms) | event | actor | summary |", markdown)


if __name__ == "__main__":
    unittest.main()
