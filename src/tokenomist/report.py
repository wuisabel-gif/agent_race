"""Rendering and export helpers for analyzer output."""

from __future__ import annotations

import csv
import io
import json

from .analyzer import AgentReport

# Columns shown in the comparison table, as (header, attribute, formatter).
_COLUMNS = [
    ("Agent", "agent", str),
    ("Model", "model", lambda v: v or "-"),
    ("Turns", "turn_count", str),
    ("→Success", "turns_to_success", lambda v: "-" if v is None else str(v)),
    ("In tok", "input_tokens", lambda v: f"{v:,}"),
    ("Out tok", "output_tokens", lambda v: f"{v:,}"),
    ("Tools", "tool_calls", str),
    ("Tool OK", "tool_success_rate", lambda v: f"{v * 100:.0f}%"),
    ("Retries", "retry_loops", str),
    ("Fixes", "correction_count", str),
    ("Latency", "latency_estimate_ms", lambda v: f"{v / 1000:.1f}s"),
    ("Cost", "cost_estimate_usd", lambda v: "n/a" if v is None else f"${v:.4f}"),
    ("Score", "final_score", lambda v: f"{v:.2f}"),
    ("Efficiency", "convergence_efficiency", lambda v: f"{v:.3f}"),
]


def rank_reports(reports: list[AgentReport]) -> list[AgentReport]:
    """Best-first ordering: convergence efficiency, then cost, then latency."""

    return sorted(
        reports,
        key=lambda r: (
            -r.convergence_efficiency,
            # Unknown-cost (n/a) reports rank after priced ones on the cost key.
            float("inf") if r.cost_estimate_usd is None else r.cost_estimate_usd,
            r.latency_estimate_ms,
        ),
    )


def render_table(reports: list[AgentReport]) -> str:
    """Render reports as a fixed-width comparison table."""

    if not reports:
        return "(no conversations to report)"

    headers = [c[0] for c in _COLUMNS]
    rows: list[list[str]] = []
    for rep in reports:
        rows.append([fmt(getattr(rep, attr)) for _, attr, fmt in _COLUMNS])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def reports_to_json(reports: list[AgentReport], *, include_trace: bool = False) -> str:
    if include_trace:
        from dataclasses import asdict

        payload = [asdict(r) for r in reports]
    else:
        payload = [r.summary_dict() for r in reports]
    return json.dumps(payload, indent=2)


def ledger_to_jsonl(
    reports: list[AgentReport],
    *,
    projection: str = "full",
    preview_chars: int = 200,
) -> str:
    """Render per-turn ledger records as JSON Lines.

    ``full`` preserves turn content. ``preview`` keeps the same accounting fields
    but truncates content for fast list views and small downstream indexes.
    """

    if projection not in {"full", "preview"}:
        raise ValueError("projection must be 'full' or 'preview'")

    lines = []
    for rep in reports:
        for row in rep.trace:
            content = row.content
            record = {
                "schema": "tokenomist.ledger.v1",
                "projection": projection,
                "record_id": f"{rep.task_id}:{rep.agent}:{row.turn_index}",
                "task_id": rep.task_id,
                "agent": rep.agent,
                "model": rep.model,
                "provider": rep.provider,
                "turn_index": row.turn_index,
                "role": row.role,
                "content_length": row.content_length,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "usage_details": row.usage_details,
                "provided_usage_details": row.provided_usage_details,
                "cost_usd": row.cost_usd,
                "cost_details": row.cost_details,
                "provided_cost_details": row.provided_cost_details,
                "latency_ms": row.latency_ms,
                "tool_calls": row.tool_calls,
                "tool_failures": row.tool_failures,
                "is_retry": row.is_retry,
                "is_correction": row.is_correction,
                "final_correct": rep.final_correct,
                "final_score": rep.final_score,
                "convergence_efficiency": rep.convergence_efficiency,
            }
            if projection == "full":
                record["content"] = content
            else:
                record["content_preview"] = _truncate_text(content, preview_chars)
                record["content_truncated"] = len(content) > preview_chars

            # Useful for spotting logs that will be expensive to store/index.
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
            record["record_bytes"] = len(encoded.encode("utf-8"))
            lines.append(json.dumps(record, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def _truncate_text(text: str, limit: int) -> str:
    if limit < 0:
        raise ValueError("preview_chars must be non-negative")
    return text if len(text) <= limit else text[:limit]


def trace_to_csv(reports: list[AgentReport]) -> str:
    """Flatten every report's per-turn trace into a single CSV document."""

    buf = io.StringIO()
    fieldnames = [
        "task_id",
        "agent",
        "model",
        "turn_index",
        "role",
        "input_tokens",
        "output_tokens",
        "usage_details",
        "provided_usage_details",
        "tool_calls",
        "tool_failures",
        "latency_ms",
        "cost_usd",
        "cost_details",
        "provided_cost_details",
        "cumulative_cost_usd",
        "is_retry",
        "is_correction",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for rep in reports:
        for row in rep.trace:
            rendered = {}
            for key in fieldnames:
                value = getattr(row, key)
                rendered[key] = (
                    json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                )
            writer.writerow(rendered)
    return buf.getvalue()
