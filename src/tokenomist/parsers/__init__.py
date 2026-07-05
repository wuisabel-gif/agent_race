"""Format detection and loading of conversation logs.

Use :func:`load_conversation` for a single file or :func:`load_conversations`
for a glob/directory. The correct parser is auto-detected from the JSON
shape; pass ``fmt=`` to force one.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

from ..models import Conversation
from .base import Parser
from .gemini import GeminiParser
from .langgraph import LangGraphParser
from .native import NativeParser
from .openai_chat import OpenAIChatParser

# Order matters: the most specific / unambiguous formats are tried first.
_PARSERS: list[Parser] = [
    NativeParser(),
    LangGraphParser(),
    OpenAIChatParser(),
    GeminiParser(),
]

_BY_NAME = {p.name: p for p in _PARSERS}


class UnknownFormatError(ValueError):
    """Raised when no registered parser recognizes a log."""


def detect_parser(data: Any) -> Parser:
    for parser in _PARSERS:
        if parser.can_parse(data):
            return parser
    raise UnknownFormatError("no registered parser recognized this log shape")


def parse_data(data: Any, fmt: str | None = None, source_path: str | None = None) -> Conversation:
    if fmt is not None:
        if fmt not in _BY_NAME:
            raise UnknownFormatError(f"unknown format {fmt!r}; known: {sorted(_BY_NAME)}")
        parser = _BY_NAME[fmt]
    else:
        parser = detect_parser(data)
    return parser.parse(data, source_path=source_path)


def load_conversation(path: str, fmt: str | None = None) -> Conversation:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_data(data, fmt=fmt, source_path=path)


def load_conversations(patterns: list[str], fmt: str | None = None) -> list[Conversation]:
    """Load every JSON file matched by ``patterns`` (globs or directories)."""

    paths: list[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            paths.extend(sorted(glob.glob(os.path.join(pattern, "*.json"))))
        else:
            matched = sorted(glob.glob(pattern))
            paths.extend(matched if matched else [pattern])

    conversations: list[Conversation] = []
    for path in paths:
        conversations.append(load_conversation(path, fmt=fmt))
    return conversations


__all__ = [
    "Parser",
    "UnknownFormatError",
    "detect_parser",
    "parse_data",
    "load_conversation",
    "load_conversations",
    "available_formats",
]


def available_formats() -> list[str]:
    return sorted(_BY_NAME)
