"""Agent orchestration framework used by this repository's demo.

Public surface:

``Orchestrator``
    planner -> researcher -> responder -> reviewer loop over support tickets.
``Agent``
    a role prompt plus an optional toolbelt and a strict JSON contract.
``OpenAICompatibleClient`` / ``MockClient``
    interchangeable model backends.
``KnowledgeBase`` / ``TicketStore``
    dependency-free data layer.
"""

from __future__ import annotations

from .agents import Agent, AgentFormatError, AgentOutcome, build_agents, extract_json
from .knowledge import Document, KnowledgeBase, SearchHit, tokenize
from .llm import Completion, LLMClient, LLMError, OpenAICompatibleClient
from .mock_llm import MockClient
from .orchestrator import Orchestrator, RunResult, summarise
from .tickets import Ticket, TicketStore
from .tools import FunctionTool, ToolRegistry, ToolResult, ToolSpec
from .trace import Trace, TraceEvent

__all__ = [
    "Agent",
    "AgentFormatError",
    "AgentOutcome",
    "Completion",
    "Document",
    "FunctionTool",
    "KnowledgeBase",
    "LLMClient",
    "LLMError",
    "MockClient",
    "OpenAICompatibleClient",
    "Orchestrator",
    "RunResult",
    "SearchHit",
    "Ticket",
    "TicketStore",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Trace",
    "TraceEvent",
    "build_agents",
    "extract_json",
    "summarise",
    "tokenize",
]

__version__ = "0.1.0"
