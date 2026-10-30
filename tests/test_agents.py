"""Tests for agent contracts and the tool loop."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

from orchestra.agents import Agent, AgentFormatError
from orchestra.llm import Completion, LLMError, Message, OpenAICompatibleClient
from orchestra.tools import FunctionTool, ToolRegistry, ToolSpec


class ScriptedClient:
    """Replays a fixed list of replies and records what it was asked."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[Sequence[Message]] = []

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> Completion:
        self.requests.append(list(messages))
        return Completion(text=self.replies.pop(0), model="scripted")


class AgentTests(unittest.TestCase):
    def test_returns_parsed_json(self) -> None:
        agent = Agent("planner", ScriptedClient(['{"priority": "high", "queries": ["x"]}']))
        outcome = agent.run("TICKET_ID: T-1\nSUBJECT: test")
        self.assertEqual(outcome.get("priority"), "high")
        self.assertEqual(outcome.tool_results, [])

    def test_runs_the_tool_loop_before_answering(self) -> None:
        registry = ToolRegistry()
        registry.register(
            FunctionTool(
                spec=ToolSpec(
                    name="echo",
                    description="echo back the value",
                    parameters={"value": "str"},
                ),
                func=lambda value: {"echoed": value},
            )
        )
        client = ScriptedClient(
            [
                '{"tool_calls": [{"name": "echo", "arguments": {"value": "hi"}}]}',
                '{"summary": "done"}',
            ]
        )
        agent = Agent("researcher", client, registry=registry)
        outcome = agent.run("TASK: say hi")
        self.assertEqual(outcome.get("summary"), "done")
        self.assertEqual(len(outcome.tool_results), 1)
        self.assertEqual(outcome.tool_results[0].output, {"echoed": "hi"})
        # The second request must carry the observation block.
        self.assertIn("OBSERVATIONS", client.requests[1][-1]["content"])

    def test_tool_request_without_registry_is_an_error(self) -> None:
        client = ScriptedClient(['{"tool_calls": [{"name": "x", "arguments": {}}]}'])
        agent = Agent("planner", client)
        with self.assertRaises(AgentFormatError):
            agent.run("TASK: whatever")

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Agent("nope", ScriptedClient([]))

    def test_context_is_rendered_as_a_block(self) -> None:
        client = ScriptedClient(['{"ok": true}'])
        Agent("responder", client).run("TASK: hi", context={"RESEARCH_NOTES": "note", "EMPTY": ""})
        content = client.requests[0][1]["content"]
        self.assertIn("CONTEXT:", content)
        self.assertIn("RESEARCH_NOTES: note", content)
        self.assertNotIn("EMPTY", content)


class ClientConfigTests(unittest.TestCase):
    def test_missing_api_key_fails_fast(self) -> None:
        client = OpenAICompatibleClient(api_key="", base_url="https://example.invalid/v1")
        with self.assertRaises(LLMError):
            client.complete([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
