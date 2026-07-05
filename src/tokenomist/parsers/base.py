"""Parser interface and helpers shared across formats."""

from __future__ import annotations

import re
from typing import Any

from ..models import Conversation, Role

# Phrases a user typically uses when correcting or redirecting an agent.
_CORRECTION_PATTERNS = re.compile(
    r"\b(no,|nope|that'?s wrong|incorrect|not what i|try again|still (broken|failing|wrong)|"
    r"doesn'?t work|that'?s not right|you (missed|forgot)|actually,|instead)\b",
    re.IGNORECASE,
)

# Phrases an assistant uses when retrying after a failure.
_RETRY_PATTERNS = re.compile(
    r"\b(let me try again|let me retry|apologies|sorry|my mistake|i'?ll fix|"
    r"let me correct|that didn'?t work|let me re-?run)\b",
    re.IGNORECASE,
)


def looks_like_correction(text: str) -> bool:
    return bool(_CORRECTION_PATTERNS.search(text or ""))


def looks_like_retry(text: str) -> bool:
    return bool(_RETRY_PATTERNS.search(text or ""))


def normalize_role(raw: str) -> Role:
    """Map a vendor role string onto the normalized :class:`Role`."""

    key = (raw or "").lower()
    mapping = {
        "user": Role.USER,
        "human": Role.USER,
        "assistant": Role.ASSISTANT,
        "ai": Role.ASSISTANT,
        "model": Role.ASSISTANT,
        "bot": Role.ASSISTANT,
        "tool": Role.TOOL,
        "function": Role.TOOL,
        "tool_result": Role.TOOL,
        "system": Role.SYSTEM,
        "developer": Role.SYSTEM,
    }
    return mapping.get(key, Role.ASSISTANT)


def coerce_text(content: Any) -> str:
    """Flatten the many shapes ``content`` takes across formats into a string.

    Handles plain strings, OpenAI-style content-part lists, and Gemini-style
    ``parts`` lists.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                chunks.append(str(part.get("text") or part.get("content") or ""))
        return "\n".join(c for c in chunks if c)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


class Parser:
    """Base class for format parsers."""

    name = "base"

    def can_parse(self, data: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def parse(self, data: Any, source_path: str | None = None) -> Conversation:  # pragma: no cover
        raise NotImplementedError
