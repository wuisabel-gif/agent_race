"""Tests for trace construction and aggregate analysis."""

from __future__ import annotations

import os

from tokenomist.analyzer import analyze, analyze_many, build_trace
from tokenomist.models import Role
from tokenomist.parsers import load_conversation, load_conversations
from tokenomist.report import rank_reports, render_table, reports_to_json, trace_to_csv

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
    csv_text = trace_to_csv(reports)
    assert "cumulative_cost_usd" in csv_text
    # One header line + at least one row per turn across all conversations.
    assert len(csv_text.strip().splitlines()) > len(reports)


def test_empty_table():
    assert "no conversations" in render_table([])
