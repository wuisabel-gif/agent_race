"""Tests for format detection and the individual parsers."""

from __future__ import annotations

import os

import pytest

from tokenomist.models import Role
from tokenomist.parsers import (
    UnknownFormatError,
    available_formats,
    detect_parser,
    load_conversations,
    parse_data,
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")


def test_available_formats():
    assert set(available_formats()) == {"native", "openai_chat", "gemini", "langgraph"}


def test_detect_native():
    data = {"turns": [{"role": "user", "content": "hi"}]}
    assert detect_parser(data).name == "native"


def test_detect_openai_vs_langgraph():
    openai = {"messages": [{"role": "user", "content": "hi"}]}
    langgraph = {"messages": [{"type": "human", "content": "hi"}]}
    assert detect_parser(openai).name == "openai_chat"
    assert detect_parser(langgraph).name == "langgraph"


def test_detect_gemini():
    data = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    assert detect_parser(data).name == "gemini"


def test_unknown_format_raises():
    with pytest.raises(UnknownFormatError):
        detect_parser({"something": "else"})


def test_force_format():
    data = {"turns": [{"role": "user", "content": "hi"}]}
    conv = parse_data(data, fmt="native")
    assert conv.source_format == "native"
    with pytest.raises(UnknownFormatError):
        parse_data(data, fmt="does-not-exist")


def test_gemini_function_calls_parsed():
    data = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"text": "calling tool"},
                    {"functionCall": {"name": "run_code", "args": {"code": "x"}}},
                    {"functionResponse": {"name": "run_code", "response": "ok"}},
                ],
            }
        ]
    }
    conv = parse_data(data)
    turn = conv.turns[0]
    assert turn.role is Role.ASSISTANT
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "run_code"
    assert turn.tool_calls[0].result == "ok"


def test_openai_string_arguments_decoded():
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "python", "arguments": '{"code": "print(1)"}'}}
                ],
            }
        ]
    }
    conv = parse_data(data)
    call = conv.turns[0].tool_calls[0]
    assert call.name == "python"
    assert call.arguments == {"code": "print(1)"}


def test_correction_and_retry_detection():
    data = {
        "turns": [
            {"role": "user", "content": "No, that's wrong, try again"},
            {"role": "assistant", "content": "Sorry, my mistake, let me retry"},
        ]
    }
    conv = parse_data(data)
    assert conv.turns[0].is_correction is True
    assert conv.turns[1].is_retry is True


def test_load_all_samples():
    convs = load_conversations([SAMPLE_DIR])
    assert len(convs) == 5
    agents = {c.agent for c in convs}
    assert {"Claude", "ChatGPT", "Gemini", "Custom Agent", "LangGraph Agent"} == agents
    assert all(c.task_id == "fix-parse-duration" for c in convs)
