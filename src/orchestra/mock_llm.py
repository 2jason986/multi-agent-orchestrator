"""A deterministic stand-in for a real language model.

Every agent in this project speaks to the model through the ``LLMClient``
protocol, so a scripted client is enough to exercise the full orchestration
loop: heuristics are not the point, reproducibility is. The mock exists so that

* ``python -m orchestra.cli demo`` works with no API key and no network,
* the test suite asserts on exact orchestration behaviour instead of vibes,
* CI stays fast and free.

Agents address the model with a ``[ROLE:<name>]`` marker in the system prompt
and pass structured context blocks in the user turn. The mock keys off both.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .llm import Completion, Message

ROLE_PATTERN = re.compile(r"\[ROLE:([a-z_]+)\]")

# Ticket keywords that should lift priority. Kept intentionally small.
URGENT_HINTS = ("宕机", "生产环境", "无法登录", "付款", "扣费", "outage", "blocked", "urgent")


def _role_of(messages: Sequence[Message]) -> str:
    for message in messages:
        if message.get("role") == "system":
            found = ROLE_PATTERN.search(message.get("content", ""))
            if found:
                return found.group(1)
    return "unknown"


def _field(text: str, label: str) -> str:
    """Pull ``LABEL: value`` out of a context block, tolerating multi-line values."""
    pattern = re.compile(rf"^{label}:\s*(.+?)(?=^[A-Z_]+:|\Z)", re.S | re.M)
    found = pattern.search(text)
    return found.group(1).strip() if found else ""


def _queries_from(text: str) -> list[str]:
    subject = _field(text, "SUBJECT") or _field(text, "TASK")
    body = _field(text, "BODY")
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", subject + " " + body)
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:3] or ["产品使用说明"]


def _plan_subject(text: str) -> tuple[str, list[str]]:
    subject = _field(text, "SUBJECT") or _field(text, "TASK")
    queries = _queries_from(text)
    if not subject and queries:
        subject = queries[0]
    return subject, queries


def _planner(messages: Sequence[Message]) -> str:
    text = messages[-1].get("content", "")
    subject, queries = _plan_subject(text)
    urgent = any(hint in text for hint in URGENT_HINTS)
    payload = {
        "priority": "high" if urgent else "normal",
        "queries": queries,
        "plan": [
            f"检索知识库：{queries[0]}",
            "依据检索结果起草回复",
            "提交审核，未通过则按反馈修订",
        ],
        "subject": subject,
    }
    return json.dumps(payload, ensure_ascii=False)


def _researcher(messages: Sequence[Message]) -> str:
    text = messages[-1].get("content", "")
    if "OBSERVATIONS:" in text:
        observations = text.split("OBSERVATIONS:", 1)[1].strip()
        try:
            data = json.loads(observations)
        except json.JSONDecodeError:
            data = {}
        hits = data.get("kb_search", {})
        results = hits.get("results", []) if isinstance(hits, dict) else []
        snippets = []
        citations = []
        for item in results[:2]:
            doc_id = item.get("doc_id", "unknown")
            snippets.append(f"- [{doc_id}] {item.get('title', '')}：{item.get('excerpt', '')}")
            if item.get("doc_id"):
                citations.append(item["doc_id"])
        summary = "知识库要点：\n" + ("\n".join(snippets) if snippets else "未命中相关文档。")
        return json.dumps(
            {"summary": summary, "citations": citations, "hit_count": len(results)},
            ensure_ascii=False,
        )

    subject, queries = _plan_subject(text)
    query = _field(text, "QUERY") or (queries[0] if queries else subject)
    return json.dumps(
        {"tool_calls": [{"name": "kb_search", "arguments": {"query": query, "top_k": 3}}]},
        ensure_ascii=False,
    )


def _responder(messages: Sequence[Message]) -> str:
    text = messages[-1].get("content", "")
    notes = _field(text, "RESEARCH_NOTES")
    feedback = _field(text, "REVIEW_FEEDBACK")
    subject = _field(text, "SUBJECT") or "您反馈的问题"
    bullets = [
        line.strip()[2:].strip()
        for line in notes.splitlines()
        if line.strip().startswith("-")
    ][:2]
    body = "\n".join(f"· {item}" for item in bullets) or "已转交人工同事进一步确认。"
    closing = "如果以上信息没有解决您的问题，请直接回复本邮件，我们会安排工程师跟进。"
    if feedback:
        closing = (
            "另外，麻烦您补充一下出现问题时使用的账号和环境信息，方便我们进一步定位。\n"
            + closing
        )

    draft = f"您好，关于「{subject}」：\n\n{body}\n\n{closing}\n"
    citations = re.findall(r"\[(kb-[0-9a-z\-]+)\]", notes)
    return json.dumps({"draft": draft, "citations": sorted(set(citations))}, ensure_ascii=False)


def _reviewer(messages: Sequence[Message]) -> str:
    text = messages[-1].get("content", "")
    draft = _field(text, "DRAFT")
    round_no = _field(text, "REVIEW_ROUND")
    citations_present = "[kb-" in text or "kb-" in draft
    problems = []
    if not citations_present:
        problems.append("回复中没有引用知识库条目，无法溯源")
    if len(draft) < 80:
        problems.append("回复内容过短，缺少可执行的处理建议")
    if round_no.strip() in {"1", ""}:
        problems.append("缺少复现步骤确认，建议先询问用户环境信息")
    if problems:
        return json.dumps(
            {"verdict": "revise", "feedback": "；".join(problems)}, ensure_ascii=False
        )
    return json.dumps(
        {"verdict": "approve", "feedback": "引用完整、结构清晰，可以发送。"}, ensure_ascii=False
    )


_HANDLERS = {
    "planner": _planner,
    "researcher": _researcher,
    "responder": _responder,
    "reviewer": _reviewer,
}


class MockClient:
    """Scripted model used for offline demos and assertions in tests."""

    name = "mock"

    def __init__(self, handlers: dict[str, object] | None = None) -> None:
        self._handlers = {**_HANDLERS, **(handlers or {})}

    def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> Completion:
        role = _role_of(messages)
        handler = self._handlers.get(role)
        if handler is None:
            return Completion(text=json.dumps({"note": f"no script for role {role}"}))
        return Completion(text=str(handler(messages)), model="mock")
