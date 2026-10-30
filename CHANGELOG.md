# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Provider presets for OpenAI, DeepSeek, Moonshot and DashScope. `--provider deepseek`
  alone is enough to run against the DeepSeek API; `custom` plus
  `ORCHESTRA_BASE_URL` covers vLLM, Ollama and internal gateways.
- Retries with exponential backoff for 408/409/425/429, 5xx and network failures.
  Auth and payload errors still fail on the first attempt.
- `--timeout` and `--json-mode` (`response_format=json_object`, honoured by OpenAI,
  DeepSeek and DashScope).
- Token usage from the API response is aggregated across agents and shown in the
  report and the `--json` summary; the mock backend reports none, so its output is
  unchanged.
- Wire-level tests that run the client against a local fake `/chat/completions`
  endpoint: request shape, retries, error unwrapping and presets.

### Fixed

- An agent that ran out of tool rounds used to hand its dangling `tool_calls`
  payload back as if it were an answer. It now drops the request, sets
  `tool_budget_exhausted` and the orchestrator records a trace warning.
- Tool calls carrying fields the tool never declared raised `TypeError` and burned
  a round; undeclared arguments are now dropped before dispatch.
- `orchestra demo` mutated ticket status in memory only; the resulting store is
  now written next to the report.

### Changed

- HTTP failures are no longer treated as one class: retryable problems raise
  `LLMRetryableError`, terminal ones raise `LLMError`.
- Provider error bodies are unwrapped to `error.message` instead of a 500-character
  JSON dump.
- Test suite grew to 51 cases.

## [0.1.0] - 2026-10-30

### Added

- Four-role orchestration loop: planner, researcher, responder, reviewer, with a
  bounded revision budget and automatic escalation to a human.
- Strict JSON output contract per agent, with a tolerant extractor for fenced and
  prose-wrapped model replies.
- Tool registry with uniform error handling and two built-in tools, `kb_search`
  and `ticket_lookup`.
- Dependency-free TF-IDF retriever over a Markdown knowledge base, including CJK
  bigram tokenisation and query-aware excerpt extraction.
- Deterministic mock model so the full pipeline runs offline, in tests and in CI.
- OpenAI-compatible model client built on `urllib`, configurable through
  `ORCHESTRA_API_KEY`, `ORCHESTRA_BASE_URL` and `ORCHESTRA_MODEL`.
- Execution tracing written as Markdown and JSON, covering agent spans, tool calls
  and review verdicts.
- CLI with `run` and `demo` subcommands.
- 34 unit and end-to-end tests running against the mock model.
- GitHub Actions CI on Python 3.10, 3.11 and 3.12, plus Dependabot, issue and pull
  request templates, Code of Conduct, security policy and contributing guide.

[Unreleased]: https://github.com/2jason986/multi-agent-orchestrator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/2jason986/multi-agent-orchestrator/releases/tag/v0.1.0
