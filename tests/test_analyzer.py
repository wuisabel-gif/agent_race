"""Tests for trace construction and aggregate analysis."""

from __future__ import annotations

import json
import os

from tokenomist.analyzer import analyze, analyze_many, build_trace
from tokenomist.models import Role
from tokenomist.parsers import load_conversation, load_conversations, parse_data
from tokenomist.report import (
    ledger_to_jsonl,
    rank_reports,
    render_table,
    reports_to_json,
    trace_to_csv,
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")


def _claude():
    return load_conversation(os.path.join(SAMPLE_DIR, "claude_native.json"))


def test_trace_input_tokens_accumulate():
    conv = _claude()
    trace = build_trace(conv)
    assistant_rows = [r for r in trace if r.role == Role.ASSISTANT.value]
    inputs = [r.input_tokens for r in assistant_rows]
    # Context grows, so each later assistant call sees at least as much input.
    assert inputs == sorted(inputs)
    assert inputs[-1] > inputs[0]


def test_cumulative_cost_monotonic():
    trace = build_trace(_claude())
    costs = [r.cumulative_cost_usd for r in trace]
    assert costs == sorted(costs)
    assert costs[-1] > 0


def test_report_fields_for_successful_run():
    rep = analyze(_claude())
    assert rep.agent == "Claude"
    assert rep.final_correct is True
    assert rep.success_turn == 4
    assert rep.turns_to_success is not None
    assert rep.tokens_to_success is not None
    assert rep.tool_calls >= 3
    assert rep.tool_success_rate == 1.0
    assert 0.0 < rep.convergence_efficiency <= 1.0


def test_non_converged_has_zero_efficiency():
    conv = load_conversation(os.path.join(SAMPLE_DIR, "gemini_google.json"))
    rep = analyze(conv)
    assert rep.success_turn is None
    assert rep.convergence_efficiency == 0.0
    assert rep.correction_count >= 2
    assert rep.retry_loops >= 1


def test_summary_dict_excludes_trace():
    rep = analyze(_claude())
    assert "trace" not in rep.summary_dict()


def test_ranking_and_table():
    reports = analyze_many(load_conversations([SAMPLE_DIR]))
    ranked = rank_reports(reports)
    # Best convergence efficiency comes first.
    assert ranked[0].convergence_efficiency == max(r.convergence_efficiency for r in reports)
    table = render_table(ranked)
    assert "Agent" in table and "Efficiency" in table
    assert "Claude" in table


def test_json_and_csv_export():
    reports = analyze_many(load_conversations([SAMPLE_DIR]))
    js = reports_to_json(reports)
    assert "convergence_efficiency" in js
    assert "usage_details" in js
    csv_text = trace_to_csv(reports)
    assert "cumulative_cost_usd" in csv_text
    assert "usage_details" in csv_text
    # One header line + at least one row per turn across all conversations.
    assert len(csv_text.strip().splitlines()) > len(reports)


def test_ledger_jsonl_full_and_preview_exports():
    reports = analyze_many(load_conversations([SAMPLE_DIR]))

    full = ledger_to_jsonl(reports, projection="full")
    first_full = json.loads(full.splitlines()[0])
    assert first_full["schema"] == "tokenomist.ledger.v1"
    assert first_full["projection"] == "full"
    assert "content" in first_full
    assert first_full["content_length"] == len(first_full["content"])
    assert first_full["record_bytes"] > 0

    preview = ledger_to_jsonl(reports, projection="preview", preview_chars=5)
    first_preview = json.loads(preview.splitlines()[0])
    assert first_preview["projection"] == "preview"
    assert "content" not in first_preview
    assert len(first_preview["content_preview"]) <= 5
    assert first_preview["content_truncated"] == (first_preview["content_length"] > 5)


def test_usage_and_cost_detail_maps_are_aggregated():
    conv = load_conversation(os.path.join(SAMPLE_DIR, "custom_agent_native.json"))
    conv.turns[1].usage_details = {
        "input_tokens": 1000,
        "cached_input_tokens": 200,
        "output_tokens": 100,
    }
    conv.turns[1].provided_usage_details = {
        "input_tokens": 1001,
        "cached_input_tokens": 201,
        "output_tokens": 99,
    }

    rep = analyze(conv)
    assert rep.usage_details["cached_input_tokens"] == 200
    assert rep.provided_usage_details["cached_input_tokens"] == 201
    assert set(rep.cost_details) >= {"input", "output", "cache_read"}


def test_provider_cost_details_are_authoritative():
    conv = parse_data(
        {
            "agent": "ProviderCostAgent",
            "model": "gpt-4o",
            "turns": [
                {
                    "role": "assistant",
                    "content": "done",
                    "usage_details": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 1_000_000,
                    },
                    "provided_cost_details": {"input": 0.01, "output": 0.02},
                }
            ],
        }
    )

    rep = analyze(conv)
    assert rep.cost_details["input"] == 0.01
    assert rep.cost_details["output"] == 0.02
    assert rep.cost_details["total"] == 0.03
    assert rep.cost_estimate_usd == 0.03


def test_empty_table():
    assert "no conversations" in render_table([])
