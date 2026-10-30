"""Tests for the tool registry and the JSON extraction helper."""

from __future__ import annotations

import unittest

from orchestra.agents import AgentFormatError, extract_json
from orchestra.tools import FunctionTool, ToolRegistry, ToolSpec


def _double(value: int) -> int:
    return value * 2


def _explode() -> None:
    raise ValueError("boom")


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(
            FunctionTool(
                spec=ToolSpec(
                    name="double",
                    description="double a number",
                    parameters={"value": "int"},
                ),
                func=_double,
            )
        )

    def test_call_returns_output(self) -> None:
        result = self.registry.call("double", {"value": 21})
        self.assertTrue(result.ok)
        self.assertEqual(result.output, 42)

    def test_unknown_tool_is_reported_not_raised(self) -> None:
        result = self.registry.call("missing", {})
        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.error or "")

    def test_tool_exception_is_captured(self) -> None:
        self.registry.register(
            FunctionTool(spec=ToolSpec(name="explode", description="always fails"), func=_explode)
        )
        result = self.registry.call("explode", {})
        self.assertFalse(result.ok)
        self.assertIn("boom", result.error or "")

    def test_duplicate_registration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.register(
                FunctionTool(spec=ToolSpec(name="double", description="again"), func=_double)
            )

    def test_describe_lists_specs(self) -> None:
        self.assertIn("double(value)", self.registry.describe())

    def test_result_serialises_for_observations(self) -> None:
        payload = self.registry.call("double", {"value": 2}).to_dict()
        self.assertEqual(payload["results"], [4])
        self.assertEqual(payload["name"], "double")


class ExtractJsonTests(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(extract_json('{"verdict": "approve"}'), {"verdict": "approve"})

    def test_object_wrapped_in_prose_and_fences(self) -> None:
        text = 'Sure, here it is:\n```json\n{"a": 1, "b": {"c": 2}}\n```\nThanks!'
        self.assertEqual(extract_json(text), {"a": 1, "b": {"c": 2}})

    def test_braces_inside_strings_do_not_confuse_the_parser(self) -> None:
        text = '{"note": "use {braces} carefully", "ok": true}'
        self.assertEqual(extract_json(text)["ok"], True)

    def test_no_json_raises(self) -> None:
        with self.assertRaises(AgentFormatError):
            extract_json("I cannot help with that.")

    def test_non_object_json_is_rejected(self) -> None:
        with self.assertRaises(AgentFormatError):
            extract_json("[1, 2, 3]")


if __name__ == "__main__":
    unittest.main()
