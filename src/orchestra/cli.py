"""Command line entry point: ``orchestra run`` and ``orchestra demo``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .knowledge import KnowledgeBase
from .llm import LLMClient, OpenAICompatibleClient
from .mock_llm import MockClient
from .orchestrator import Orchestrator, RunResult, summarise
from .tickets import TicketStore


def _resolve_data_dir(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path("data"))
    candidates.append(Path(__file__).resolve().parents[2] / "data")
    for candidate in candidates:
        if (candidate / "knowledge_base").is_dir() and (candidate / "tickets.json").is_file():
            return candidate
    raise SystemExit(
        "Cannot find the data directory. Pass --data-dir pointing at a folder that "
        "contains knowledge_base/ and tickets.json."
    )


def _build_client(provider: str, model: str | None) -> LLMClient:
    if provider == "mock":
        return MockClient()
    return OpenAICompatibleClient(model=model)


def _build_orchestrator(args: argparse.Namespace) -> Orchestrator:
    data_dir = _resolve_data_dir(args.data_dir)
    return Orchestrator(
        llm=_build_client(args.provider, args.model),
        knowledge_base=KnowledgeBase.from_directory(data_dir / "knowledge_base"),
        ticket_store=TicketStore.from_json(data_dir / "tickets.json"),
        max_review_rounds=args.max_review_rounds,
    )


def _write_report(results: list[RunResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)
    lines = [
        "# Orchestration demo report",
        "",
        f"- tickets processed: {summary['tickets']}",
        f"- approved by reviewer: {summary['approved']}",
        f"- escalated to a human: {summary['needs_human']}",
        f"- review rounds in total: {summary['rounds']}",
        "",
    ]
    for result in results:
        lines.append(result.to_markdown())
        lines.append("---")
        lines.append("")
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    traces = [result.trace.to_dict() for result in results if result.trace]
    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def cmd_run(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator(args)
    ticket = orchestrator.ticket_store.get(args.ticket)
    result = orchestrator.handle(ticket)
    print(result.to_markdown())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(result.to_markdown(), encoding="utf-8")
        print(f"\nwritten to {args.out}")
    if args.json:
        payload = {
            "ticket_id": result.ticket_id,
            "approved": result.approved,
            "rounds": result.rounds,
            "priority": result.priority,
            "citations": result.citations,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator(args)
    results = orchestrator.handle_all()
    if not results:
        print("No open tickets found.")
        return 0
    summary = summarise(results)
    print(f"provider      : {args.provider}")
    print(f"tickets       : {summary['tickets']}")
    print(f"approved      : {summary['approved']}")
    print(f"needs a human : {summary['needs_human']}")
    print(f"review rounds : {summary['rounds']}")
    report = _write_report(results, Path(args.out))
    print(f"report        : {report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestra",
        description="A multi-agent orchestrator that answers support tickets.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openai"],
        default="mock",
        help="model backend: mock (offline, default) or any OpenAI-compatible endpoint",
    )
    parser.add_argument("--model", default=None, help="model name, e.g. deepseek-chat")
    parser.add_argument(
        "--data-dir", default=None, help="folder holding knowledge_base/ and tickets.json"
    )
    parser.add_argument(
        "--max-review-rounds", type=int, default=2, help="revision budget per ticket"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="handle a single ticket")
    run.add_argument("--ticket", required=True, help="ticket id, e.g. T-1001")
    run.add_argument("--out", default=None, help="write the markdown report to this file")
    run.add_argument("--json", action="store_true", help="also print a JSON summary")
    run.set_defaults(func=cmd_run)

    demo = sub.add_parser("demo", help="handle every open ticket and write a report")
    demo.add_argument("--out", default="examples/output", help="output folder")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
