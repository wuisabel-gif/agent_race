"""Tokenomist — characterize and compare agentic LLM workloads.

Converts multi-turn AI conversations (ChatGPT, Gemini, Claude, OpenAI Agents
SDK, LangGraph, or custom logs) into structured traffic traces, then estimates
latency, token cost, tool-call patterns, retries, and convergence efficiency
so different agent systems can be ranked on the same task.
"""

from __future__ import annotations

from .analyzer import AgentReport, TraceRow, analyze, analyze_many, build_trace
from .models import Conversation, Role, ToolCall, Turn
from .parsers import load_conversation, load_conversations, parse_data
from .pricing import PriceBook

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AgentReport",
    "TraceRow",
    "Conversation",
    "Turn",
    "ToolCall",
    "Role",
    "PriceBook",
    "analyze",
    "analyze_many",
    "build_trace",
    "load_conversation",
    "load_conversations",
    "parse_data",
]
