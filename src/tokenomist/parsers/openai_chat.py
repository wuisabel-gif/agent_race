"""Parser for OpenAI / ChatGPT-style message lists.

Matches the Chat Completions request/response shape and the OpenAI Agents
SDK transcript shape: a ``messages`` list of ``{"role", "content"}`` objects,
where assistant messages may carry ``tool_calls`` and tool outputs arrive as
messages with ``role == "tool"``.
"""

from __future__ import annotations

import json
from typing import Any

from ..models import Conversation, Role, ToolCall, Turn
from .base import Parser, coerce_text, looks_like_correction, looks_like_retry, normalize_role


class OpenAIChatParser(Parser):
    name = "openai_chat"

    def can_parse(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        first = messages[0]
        return isinstance(first, dict) and "role" in first

    def _parse_tool_calls(self, raw: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for tc in raw.get("tool_calls", []) or []:
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {"_raw": args}
            calls.append(
                ToolCall(
                    name=str(fn.get("name", "tool")),
                    arguments=args if isinstance(args, dict) else {"_raw": args},
                    ok=bool(tc.get("ok", True)),
                    result=tc.get("result"),
                )
            )
        return calls

    def parse(self, data: Any, source_path: str | None = None) -> Conversation:
        turns: list[Turn] = []
        for i, raw in enumerate(data["messages"]):
            role = normalize_role(raw.get("role", "assistant"))
            text = coerce_text(raw.get("content"))
            turns.append(
                Turn(
                    index=i,
                    role=role,
                    content=text,
                    input_tokens=raw.get("input_tokens"),
                    output_tokens=raw.get("output_tokens"),
                    tool_calls=self._parse_tool_calls(raw),
                    latency_ms=raw.get("latency_ms"),
                    is_correction=role is Role.USER and looks_like_correction(text),
                    is_retry=role is Role.ASSISTANT and looks_like_retry(text),
                )
            )

        return Conversation(
            agent=str(data.get("agent", "ChatGPT")),
            turns=turns,
            task_id=str(data.get("task_id", "task")),
            model=data.get("model", "gpt-4o"),
            provider=data.get("provider", "openai"),
            success_turn=data.get("success_turn"),
            final_correct=bool(data.get("final_correct", False)),
            final_score=float(data.get("final_score", 0.0)),
            source_format=self.name,
            source_path=source_path,
        )
