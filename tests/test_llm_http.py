"""Wire-level tests for the OpenAI-compatible client.

A throwaway local endpoint replays canned replies, so these tests prove the
request shape DeepSeek and friends expect without a network or an API key.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import patch

from orchestra.llm import LLMError, LLMRetryableError, OpenAICompatibleClient

CHAT_REPLY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"verdict": "approve"}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 31, "completion_tokens": 7, "total_tokens": 38},
}


class FakeEndpoint:
    """A local ``/chat/completions`` endpoint that replays canned responses."""

    def __init__(self, replies: list[tuple[int, dict[str, Any]]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                endpoint.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                status, payload = endpoint.replies.pop(0)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> FakeEndpoint:
        self.thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _client(endpoint: FakeEndpoint, **overrides: Any) -> OpenAICompatibleClient:
    options: dict[str, Any] = {
        "api_key": "sk-test",
        "base_url": endpoint.base_url,
        "retry_backoff": 0.0,
    }
    options.update(overrides)
    return OpenAICompatibleClient(**options)


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "TICKET_ID: T-1001"}]


class CompletionTests(unittest.TestCase):
    def test_openai_shaped_reply_is_parsed(self) -> None:
        with FakeEndpoint([(200, CHAT_REPLY)]) as endpoint:
            completion = _client(endpoint).complete(_messages())
        self.assertEqual(completion.text, '{"verdict": "approve"}')
        self.assertEqual(completion.model, "deepseek-chat")
        self.assertEqual(completion.usage["prompt_tokens"], 31)
        self.assertEqual(completion.usage["total_tokens"], 38)

    def test_request_matches_the_chat_completions_contract(self) -> None:
        with FakeEndpoint([(200, CHAT_REPLY)]) as endpoint:
            _client(endpoint, provider="deepseek").complete(_messages(), temperature=0.2)
            request = endpoint.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["authorization"], "Bearer sk-test")
        self.assertEqual(request["body"]["model"], "deepseek-chat")
        self.assertEqual(request["body"]["messages"], _messages())
        self.assertEqual(request["body"]["temperature"], 0.2)
        self.assertFalse(request["body"]["stream"])
        self.assertNotIn("response_format", request["body"])

    def test_json_mode_asks_the_server_for_a_json_object(self) -> None:
        with FakeEndpoint([(200, CHAT_REPLY)]) as endpoint:
            _client(endpoint, json_mode=True).complete(_messages())
            body = endpoint.requests[0]["body"]
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_missing_api_key_fails_before_any_request(self) -> None:
        client = OpenAICompatibleClient(api_key="", base_url="https://example.invalid/v1")
        with patch.dict(os.environ, {"ORCHESTRA_API_KEY": ""}):
            with self.assertRaises(LLMError):
                client.complete(_messages())


class RetryTests(unittest.TestCase):
    def test_rate_limit_is_retried_then_succeeds(self) -> None:
        replies = [(429, {"error": {"message": "Rate limit reached"}}), (200, CHAT_REPLY)]
        with FakeEndpoint(replies) as endpoint:
            completion = _client(endpoint, max_retries=1).complete(_messages())
            self.assertEqual(len(endpoint.requests), 2)
        self.assertEqual(completion.text, '{"verdict": "approve"}')

    def test_retry_budget_is_bounded(self) -> None:
        replies = [
            (503, {"error": {"message": "upstream busy"}}),
            (503, {"error": {"message": "upstream busy"}}),
        ]
        with FakeEndpoint(replies) as endpoint:
            with self.assertRaises(LLMRetryableError):
                _client(endpoint, max_retries=1).complete(_messages())
            self.assertEqual(len(endpoint.requests), 2)

    def test_auth_failure_is_not_retried(self) -> None:
        replies = [(401, {"error": {"message": "Authentication Fails"}})]
        with FakeEndpoint(replies) as endpoint:
            with self.assertRaises(LLMError) as caught:
                _client(endpoint, max_retries=2).complete(_messages())
            self.assertEqual(len(endpoint.requests), 1)
        message = str(caught.exception)
        self.assertIn("Authentication Fails", message)
        self.assertIn("401", message)
        self.assertNotIsInstance(caught.exception, LLMRetryableError)

    def test_connection_refused_is_reported_as_retryable(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = OpenAICompatibleClient(
            api_key="sk-test",
            base_url=f"http://127.0.0.1:{port}/v1",
            max_retries=0,
            retry_backoff=0.0,
        )
        with self.assertRaises(LLMRetryableError):
            client.complete(_messages())


class PresetTests(unittest.TestCase):
    def test_deepseek_preset_fills_base_url_and_model(self) -> None:
        with patch.dict(os.environ, {"ORCHESTRA_BASE_URL": "", "ORCHESTRA_MODEL": ""}):
            client = OpenAICompatibleClient(provider="deepseek", api_key="sk-test")
        self.assertEqual(client.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(client.model, "deepseek-chat")
        self.assertEqual(client.name, "deepseek:deepseek-chat")

    def test_environment_overrides_the_preset(self) -> None:
        env = {"ORCHESTRA_BASE_URL": "http://localhost:8000/v1", "ORCHESTRA_MODEL": "local"}
        with patch.dict(os.environ, env):
            client = OpenAICompatibleClient(provider="deepseek", api_key="sk-test")
        self.assertEqual(client.base_url, "http://localhost:8000/v1")
        self.assertEqual(client.model, "local")

    def test_custom_provider_falls_back_to_the_openai_default(self) -> None:
        with patch.dict(os.environ, {"ORCHESTRA_BASE_URL": "", "ORCHESTRA_MODEL": ""}):
            client = OpenAICompatibleClient(provider="custom", api_key="sk-test")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.model, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
