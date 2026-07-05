"""Parser for Tokenomist's own normalized JSON schema.

This is the richest input format: it carries explicit agent/model metadata,
per-turn token counts, tool calls, and ground-truth success/score fields. It
is also what every other parser conceptually maps onto. See
``data/samples`` for examples.
"""

from __future__ import annotations

from typing import Any

from ..models import Conversation, Role, ToolCall, Turn
from .base import Parser, coerce_text, looks_like_correction, looks_like_retry, normalize_role


class NativeParser(Parser):
    name = "native"

    def can_parse(self, data: Any) -> bool:
        return isinstance(data, dict) and "turns" in data and isinstance(data["turns"], list)

    def parse(self, data: Any, source_path: str | None = None) -> Conversation:
        turns: list[Turn] = []
        for i, raw in enumerate(data["turns"]):
            role = normalize_role(raw.get("role", "assistant"))
            text = coerce_text(raw.get("content"))
            tool_calls = [ToolCall.from_dict(tc) for tc in raw.get("tool_calls", []) or []]
            turn = Turn(
                index=i,
                role=role,
                content=text,
                input_tokens=raw.get("input_tokens"),
                output_tokens=raw.get("output_tokens"),
                tool_calls=tool_calls,
                latency_ms=raw.get("latency_ms"),
                timestamp=raw.get("timestamp"),
                is_correction=bool(
                    raw.get("is_correction", role is Role.USER and looks_like_correction(text))
                ),
                is_retry=bool(
                    raw.get("is_retry", role is Role.ASSISTANT and looks_like_retry(text))
                ),
            )
            turns.append(turn)

        return Conversation(
            agent=str(data.get("agent", "Agent")),
            turns=turns,
            task_id=str(data.get("task_id", "task")),
            model=data.get("model"),
            provider=data.get("provider"),
            success_turn=data.get("success_turn"),
            final_correct=bool(data.get("final_correct", False)),
            final_score=float(data.get("final_score", 0.0)),
            source_format=self.name,
            source_path=source_path,
        )
