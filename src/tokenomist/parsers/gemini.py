"""Parser for Google Gemini ``generateContent``-style transcripts.

Gemini uses a ``contents`` list where each entry has a ``role`` of ``user``
or ``model`` and a ``parts`` list. Function calls appear as parts with a
``functionCall`` key and results as ``functionResponse``.
"""

from __future__ import annotations

from typing import Any

from ..models import Conversation, Role, ToolCall, Turn
from .base import Parser, looks_like_correction, looks_like_retry, normalize_role


class GeminiParser(Parser):
    name = "gemini"

    def can_parse(self, data: Any) -> bool:
        return (
            isinstance(data, dict)
            and isinstance(data.get("contents"), list)
            and bool(data["contents"])
            and isinstance(data["contents"][0], dict)
            and "parts" in data["contents"][0]
        )

    def _parts_to_text_and_calls(self, parts: list[Any]) -> tuple[str, list[ToolCall]]:
        text_chunks: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if not isinstance(part, dict):
                text_chunks.append(str(part))
                continue
            if "text" in part:
                text_chunks.append(str(part["text"]))
            if "functionCall" in part:
                fc = part["functionCall"] or {}
                calls.append(
                    ToolCall(
                        name=str(fc.get("name", "tool")),
                        arguments=dict(fc.get("args", {}) or {}),
                    )
                )
            if "functionResponse" in part:
                fr = part["functionResponse"] or {}
                # Attach the response to the most recent call when possible.
                if calls:
                    calls[-1].result = str(fr.get("response", ""))
        return "\n".join(c for c in text_chunks if c), calls

    def parse(self, data: Any, source_path: str | None = None) -> Conversation:
        turns: list[Turn] = []
        for i, raw in enumerate(data["contents"]):
            role = normalize_role(raw.get("role", "model"))
            text, calls = self._parts_to_text_and_calls(raw.get("parts", []) or [])
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
            agent=str(data.get("agent", "Gemini")),
            turns=turns,
            task_id=str(data.get("task_id", "task")),
            model=data.get("model", "gemini-1.5-pro"),
            provider=data.get("provider", "google"),
            success_turn=data.get("success_turn"),
            final_correct=bool(data.get("final_correct", False)),
            final_score=float(data.get("final_score", 0.0)),
            source_format=self.name,
            source_path=source_path,
        )
