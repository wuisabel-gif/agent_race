"""Parser for LangGraph / LangChain message-list state dumps.

LangChain serializes messages with a ``type`` discriminator
(``human`` / ``ai`` / ``tool`` / ``system``) rather than ``role``. AI
messages can carry ``tool_calls``; tool messages carry the result and the
``name`` of the tool that produced it.
"""

from __future__ import annotations

from typing import Any

from ..models import Conversation, Role, ToolCall, Turn
from .base import Parser, coerce_text, looks_like_correction, looks_like_retry, normalize_role


class LangGraphParser(Parser):
    name = "langgraph"

    def can_parse(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        first = messages[0]
        return isinstance(first, dict) and "type" in first and "role" not in first

    def parse(self, data: Any, source_path: str | None = None) -> Conversation:
        turns: list[Turn] = []
        for i, raw in enumerate(data["messages"]):
            role = normalize_role(raw.get("type", "ai"))
            text = coerce_text(raw.get("content"))
            calls = [
                ToolCall(
                    name=str(tc.get("name", "tool")),
                    arguments=dict(tc.get("args", tc.get("arguments", {})) or {}),
                )
                for tc in raw.get("tool_calls", []) or []
            ]
            turns.append(
                Turn(
                    index=i,
                    role=role,
                    content=text,
                    tool_calls=calls,
                    is_correction=role is Role.USER and looks_like_correction(text),
                    is_retry=role is Role.ASSISTANT and looks_like_retry(text),
                )
            )

        return Conversation(
            agent=str(data.get("agent", "LangGraph Agent")),
            turns=turns,
            task_id=str(data.get("task_id", "task")),
            model=data.get("model"),
            provider=data.get("provider", "langgraph"),
            success_turn=data.get("success_turn"),
            final_correct=bool(data.get("final_correct", False)),
            final_score=float(data.get("final_score", 0.0)),
            source_format=self.name,
            source_path=source_path,
        )
