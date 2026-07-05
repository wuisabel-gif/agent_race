"""Normalized data model for multi-agent conversation traces.

Every supported export format (ChatGPT, Gemini, Claude, OpenAI Agents SDK,
LangGraph, custom) is parsed into the structures defined here. Downstream
analysis only ever looks at this normalized shape, which keeps the analyzer
decoupled from the messy reality of each vendor's log format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Who produced a turn."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """A single tool/function invocation made inside an assistant turn."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    result: str | None = None
    latency_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=str(data.get("name", "unknown")),
            arguments=dict(data.get("arguments", {}) or {}),
            ok=bool(data.get("ok", True)),
            result=data.get("result"),
            latency_ms=data.get("latency_ms"),
        )


@dataclass
class Turn:
    """One message in a conversation.

    Token counts and latency are optional. When absent they are estimated by
    the analyzer, so a log with no instrumentation still yields useful numbers.
    """

    index: int
    role: Role
    content: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float | None = None
    timestamp: float | None = None
    # User turns that correct/redirect the agent; assistant turns that retry.
    is_correction: bool = False
    is_retry: bool = False

    @property
    def char_len(self) -> int:
        return len(self.content or "")


@dataclass
class Conversation:
    """A full agent run for a single task.

    ``success_turn`` is the (0-based) index of the assistant turn that first
    produced a useful/correct answer, or ``None`` if the agent never converged.
    """

    agent: str
    turns: list[Turn]
    task_id: str = "task"
    model: str | None = None
    provider: str | None = None
    success_turn: int | None = None
    final_correct: bool = False
    final_score: float = 0.0
    source_format: str = "unknown"
    source_path: str | None = None

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role is Role.ASSISTANT]

    @property
    def converged(self) -> bool:
        return self.success_turn is not None
