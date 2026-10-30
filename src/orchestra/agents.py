"""Agents: a role prompt, an optional toolbelt, and a strict output contract.

Every agent returns one JSON object. That single rule is what makes the
orchestrator able to route on ``priority``, ``tool_calls``, ``verdict`` and
other fields without parsing prose, and it is the difference between a demo
that works once and a pipeline you can put behind a test suite.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, Message
from .tools import ToolRegistry, ToolResult


class AgentFormatError(RuntimeError):
    """Raised when a model reply does not contain a usable JSON object."""


def extract_json(text: str) -> dict[str, Any]:
    """Return the first complete JSON object found in ``text``.

    Models love to wrap JSON in prose or fences, so we scan for balanced braces
    instead of trusting ``json.loads`` on the whole string.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : index + 1]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    raise AgentFormatError(f"no JSON object in model output: {text[:200]!r}")


INSTRUCTIONS: dict[str, str] = {
    "planner": (
        "You are the planning agent of a support-ticket pipeline.\n"
        "Read the ticket and decide how the team should handle it.\n"
        "Respond with a single JSON object and nothing else:\n"
        '{"priority": "high" | "normal" | "low", "queries": ["search query", "..."], '
        '"plan": ["step", "..."], "subject": "short restatement of the issue"}'
    ),
    "researcher": (
        "You are the research agent. Collect evidence from the knowledge base "
        "before anyone writes a reply.\n"
        "If you still need evidence, respond only with:\n"
        '{"tool_calls": [{"name": "kb_search", "arguments": {"query": "...", "top_k": 3}}]}\n'
        "Once you have observations, respond only with:\n"
        '{"summary": "bullet points, each prefixed with the source id in square brackets", '
        '"citations": ["kb-..."]}\n'
        "Never state a policy, price or limit that the observations do not contain."
    ),
    "responder": (
        "You are the response agent. Write the customer-facing reply in the customer's language.\n"
        "Use RESEARCH_NOTES as the only source of facts and keep citation ids in square brackets.\n"
        "If REVIEW_FEEDBACK is present, fix exactly what the reviewer asked for.\n"
        'Respond only with: {"draft": "the full reply", "citations": ["kb-..."]}'
    ),
    "reviewer": (
        "You are the review agent and you have veto power over outgoing replies.\n"
        "Check three rules:\n"
        "1. every factual claim is traceable to a cited knowledge base id,\n"
        "2. no invented policy, price or timeline,\n"
        "3. the customer is given one concrete next step.\n"
        "In the first round, ask for at least one improvement unless the draft is flawless.\n"
        'Respond only with: {"verdict": "approve" | "revise", "feedback": "one or two sentences"}'
    ),
}


@dataclass(slots=True)
class AgentOutcome:
    agent: str
    data: dict[str, Any]
    raw: str
    tool_results: list[ToolResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class Agent:
    """A role-specialised wrapper around an ``LLMClient``.

    The agent owns the conversation with the model, including the optional tool
    loop, so the orchestrator only deals with decisions and data flow.
    """

    def __init__(
        self,
        role: str,
        llm: LLMClient,
        *,
        name: str | None = None,
        registry: ToolRegistry | None = None,
        instructions: str | None = None,
        max_tool_rounds: int = 2,
    ) -> None:
        self.role = role
        self.name = name or role
        self.llm = llm
        self.registry = registry
        self.instructions = instructions or INSTRUCTIONS.get(role, "")
        if not self.instructions:
            raise ValueError(f"No instructions for role {role!r}")
        self.max_tool_rounds = max_tool_rounds

    def _user_content(self, task: str, context: dict[str, Any] | None) -> str:
        blocks = [task.strip()]
        if context:
            lines = [f"{key}: {value}" for key, value in context.items() if value not in (None, "")]
            if lines:
                blocks.append("CONTEXT:\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentOutcome:
        started = time.perf_counter()
        messages: list[Message] = [
            {"role": "system", "content": f"[ROLE:{self.role}] {self.instructions}"},
            {"role": "user", "content": self._user_content(task, context)},
        ]
        tool_results: list[ToolResult] = []
        raw = ""
        data: dict[str, Any] = {}

        for _ in range(self.max_tool_rounds + 1):
            completion = self.llm.complete(messages)
            raw = completion.text
            data = extract_json(completion.text)
            calls = data.get("tool_calls") or []
            if not calls:
                break
            if self.registry is None:
                raise AgentFormatError(f"{self.name} requested tools but has no registry")
            messages.append({"role": "assistant", "content": completion.text})
            observations: dict[str, Any] = {}
            for call in calls:
                name = str(call.get("name", ""))
                arguments = call.get("arguments") or {}
                result = self.registry.call(name, arguments)
                tool_results.append(result)
                observations[name] = result.to_dict()
            messages.append(
                {
                    "role": "user",
                    "content": "OBSERVATIONS:\n" + json.dumps(observations, ensure_ascii=False),
                }
            )

        return AgentOutcome(
            agent=self.name,
            data=data,
            raw=raw,
            tool_results=tool_results,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def build_agents(llm: LLMClient, registry: ToolRegistry) -> dict[str, Agent]:
    """Convenience factory used by the orchestrator and the tests."""
    return {
        role: Agent(role, llm, registry=registry if role == "researcher" else None)
        for role in ("planner", "researcher", "responder", "reviewer")
    }
