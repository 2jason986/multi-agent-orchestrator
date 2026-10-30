# Architecture

This document explains how a ticket actually flows through the system, where the
extension points are, and which failures the design already handles.

## Call graph

```
Orchestrator.handle(ticket)
├── Trace.record("ticket.received")
├── planner.run(prompt)
│     └── LLMClient.complete([system(role), user(ticket)])
├── researcher.run(prompt, context={QUERY})
│     ├── LLMClient.complete(...) -> {"tool_calls": [...]}
│     ├── ToolRegistry.call("kb_search", {...})
│     └── LLMClient.complete(..., OBSERVATIONS) -> {"summary", "citations"}
├── responder.run(prompt, context={RESEARCH_NOTES})
│     └── LLMClient.complete(...) -> {"draft", "citations"}
├── loop up to max_review_rounds
│     ├── reviewer.run(prompt, context={DRAFT, REVIEW_ROUND})
│     ├── approve -> break
│     └── revise  -> responder.run(..., REVIEW_FEEDBACK) -> reply replaced
└── TicketStore.set_status(answered | needs_human)
```

Two invariants make the loop predictable:

1. **The orchestrator owns all state transitions.** Agents read their context from
   the user message and write a single JSON object back. They hold no memory
   between calls, so re-running a step is safe.
2. **Every model reply is parsed before it is trusted.** `extract_json` returns a
   dict or raises `AgentFormatError`. Nothing downstream sees raw prose.

## Context blocks

Agents receive a fixed text format, not a chat history dump:

```
TICKET_ID: T-1001
CHANNEL: email
CUSTOMER: ops@acme-logistics.cn
SUBJECT: 数据导出任务一直卡在 90%
BODY:
...

CONTEXT:
QUERY: 数据导出任务一直卡在
```

Tool observations are appended as a separate turn:

```
OBSERVATIONS:
{"kb_search": {"name": "kb_search", "arguments": {...}, "results": [...]}}
```

This is deliberately boring. Structured context makes prompts diffable, makes the
mock model able to respond deterministically, and makes traces readable six months
later.

## Failure modes handled today

| Failure | Behaviour |
|---|---|
| Model returns prose instead of JSON | `extract_json` scans for a balanced object; if none, `AgentFormatError` surfaces to the CLI |
| Tool raises | `ToolRegistry.call` captures it into `ToolResult(ok=False, error=...)` and the model sees the error in the next observation |
| Tool hallucinated by the model | Unknown tool name is reported the same way, not raised |
| Tool call carries undeclared arguments | `ToolRegistry.call` drops them before dispatch, so a chatty model cannot turn a valid call into a `TypeError` |
| Agent wants more tools than its budget | The dangling `tool_calls` request is removed, `tool_budget_exhausted` is set on the outcome, and the orchestrator records an `agent.warning` event instead of mistaking the request for an answer |
| Reviewer keeps rejecting | Loop stops at `max_review_rounds`, ticket status becomes `needs_human` |
| Retrieval returns nothing | Researcher answers "no matching notes", responder is told to escalate rather than invent |
| Model endpoint unreachable or key missing | `LLMError` with the endpoint and status, printed by the CLI with exit code 1 |

## Failure modes not handled yet

- **Prompt injection from retrieved documents.** Knowledge-base text enters the
  context as-is. In production, retrieved content should be fenced and the model
  instructed to treat it as data only.
- **No retry on malformed JSON.** A single bad reply aborts the run instead of
  re-asking the model once with a correction hint.
- **No token or cost accounting.** Each agent call is timed but not priced.
- **Single-threaded.** Tickets are processed sequentially; nothing is shared, so
  parallelising at the `handle_all` level is safe but not implemented.

## Extension points

**A new agent role.** Add an entry to `INSTRUCTIONS` in `agents.py`, give it an
output contract, and call it from `Orchestrator.handle`. `build_agents` picks up
roles by name.

**A different model per role.** `Orchestrator` accepts an `agents=` mapping; build
each `Agent` with its own client, for example a cheap model for the planner and a
stronger one for the reviewer.

**A real retriever.** Implement an object with a `search(query, top_k)` method
returning `SearchHit`s and wrap it in `kb_search_tool`. Nothing else changes.

**A different domain.** The orchestration loop knows nothing about support
tickets: it only needs a task string, a role prompt and an output contract. A code
review pipeline or a document triage pipeline is the same loop with different
instructions and tools.
