"""Command-line interface for Tokenomist.

Examples
--------
    tokenomist analyze data/samples/*.json
    tokenomist calibrate runs/fix-tests
    tokenomist init-agents --out agents.json
    tokenomist route job.md --agents agents.json --out runs/job1
    tokenomist trace data/samples --csv traces.csv
    tokenomist formats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .analyzer import AgentReport, analyze_many
from .parsers import available_formats, load_conversations
from .report import rank_reports, render_table, reports_to_json, trace_to_csv
from .router import load_agents, run_terminal_agent, write_example_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON log files, globs, or directories to load.",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=available_formats(),
        default=None,
        help="Force a parser instead of auto-detecting.",
    )
    parser.add_argument(
        "--prices",
        metavar="PATH",
        default=None,
        help="Path to a custom price book JSON (see the bundled prices.json). "
        "Overrides the built-in prices so cost numbers stay current.",
    )


def _load(args: argparse.Namespace):
    conversations = load_conversations(args.paths, fmt=args.fmt)
    if not conversations:
        print("No conversations found.", file=sys.stderr)
        raise SystemExit(2)
    prices = None
    if getattr(args, "prices", None):
        from .pricing import PriceBook

        prices = PriceBook.from_file(args.prices)
    return analyze_many(conversations, prices)


def cmd_analyze(args: argparse.Namespace) -> int:
    reports = rank_reports(_load(args))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(reports_to_json(reports, include_trace=args.with_trace))
        print(f"Wrote {len(reports)} report(s) to {args.json}")
    else:
        print(render_table(reports))
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    reports = _load(args)
    csv_text = trace_to_csv(reports)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write(csv_text)
        rows = sum(len(r.trace) for r in reports)
        print(f"Wrote {rows} trace rows to {args.csv}")
    else:
        sys.stdout.write(csv_text)
    return 0


def cmd_formats(_args: argparse.Namespace) -> int:
    for name in available_formats():
        print(name)
    return 0


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _render_calibration(reports, *, min_score: float) -> str:
    by_agent = {}
    for report in reports:
        by_agent.setdefault(report.agent, []).append(report)

    rows = []
    for agent, items in by_agent.items():
        correct_items = [
            item for item in items if item.final_correct and item.final_score >= min_score
        ]
        known_cost = all(item.cost_estimate_usd is not None for item in items)
        total_cost = (
            None if not known_cost else sum(item.cost_estimate_usd or 0.0 for item in items)
        )
        cost_per_correct = (
            None if total_cost is None or not correct_items else total_cost / len(correct_items)
        )
        rows.append(
            {
                "agent": agent,
                "runs": len(items),
                "correct": len(correct_items),
                "success_rate": len(correct_items) / len(items),
                "total_cost": total_cost,
                "cost_per_correct": cost_per_correct,
                "avg_score": sum(item.final_score for item in items) / len(items),
                "avg_efficiency": sum(item.convergence_efficiency for item in items) / len(items),
                "avg_latency_ms": sum(item.latency_estimate_ms for item in items) / len(items),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["success_rate"],
            float("inf") if row["cost_per_correct"] is None else row["cost_per_correct"],
            -row["avg_efficiency"],
            row["avg_latency_ms"],
        )
    )
    headers = [
        "Agent",
        "Runs",
        "Correct",
        "Success",
        "Total cost",
        "Cost/correct",
        "Avg score",
        "Avg eff",
        "Avg latency",
    ]
    table_rows = [
        [
            row["agent"],
            str(row["runs"]),
            str(row["correct"]),
            f"{row['success_rate'] * 100:.0f}%",
            _money(row["total_cost"]),
            _money(row["cost_per_correct"]),
            f"{row['avg_score']:.2f}",
            f"{row['avg_efficiency']:.3f}",
            f"{row['avg_latency_ms'] / 1000:.1f}s",
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for idx, cell in enumerate(table_row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(cells):
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    lines = [fmt(headers), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in table_rows)
    if rows:
        winner = rows[0]
        lines.append("")
        lines.append(
            "Calibration recommendation: route this task family to "
            f"{winner['agent']} first "
            f"({winner['success_rate'] * 100:.0f}% success, "
            f"{_money(winner['cost_per_correct'])} per correct solution)."
        )
    return "\n".join(lines)


def _render_route_calibration(
    reports,
    *,
    weak_agent: str,
    strong_agent: str,
    target_pgr: float,
) -> str:
    by_task: dict[str, dict[str, AgentReport]] = {}
    for report in reports:
        if report.agent in {weak_agent, strong_agent}:
            by_task.setdefault(report.task_id, {})[report.agent] = report

    pairs = [
        (task_id, items[weak_agent], items[strong_agent])
        for task_id, items in by_task.items()
        if weak_agent in items and strong_agent in items
    ]
    if not pairs:
        return (
            "No paired runs found. Re-run both agents on the same task_id values, "
            "then pass --weak-agent and --strong-agent again."
        )

    weak_avg = sum(weak.final_score for _, weak, _ in pairs) / len(pairs)
    strong_avg = sum(strong.final_score for _, _, strong in pairs) / len(pairs)
    total_gap = sum(max(0.0, strong.final_score - weak.final_score) for _, weak, strong in pairs)
    if total_gap <= 0:
        return (
            f"{strong_agent} did not improve over {weak_agent} on the paired runs. "
            "No strong-call threshold is useful for this calibration set."
        )

    target_pgr = max(0.0, min(1.0, target_pgr))
    target_gain = target_pgr * total_gap
    candidates = sorted(
        pairs,
        key=lambda pair: (
            -(pair[2].final_score - pair[1].final_score),
            (
                float("inf")
                if pair[1].cost_estimate_usd is None or pair[2].cost_estimate_usd is None
                else pair[2].cost_estimate_usd - pair[1].cost_estimate_usd
            ),
        ),
    )

    selected = []
    recovered = 0.0
    if target_gain > 0:
        for task_id, weak, strong in candidates:
            gain = max(0.0, strong.final_score - weak.final_score)
            if gain <= 0:
                continue
            selected.append((task_id, weak, strong, gain))
            recovered += gain
            if recovered >= target_gain:
                break

    strong_call_share = len(selected) / len(pairs)
    achieved_pgr = recovered / total_gap
    routed_score = weak_avg + recovered / len(pairs)

    known_costs = all(
        weak.cost_estimate_usd is not None and strong.cost_estimate_usd is not None
        for _, weak, strong in pairs
    )
    if known_costs:
        all_weak_cost = sum(weak.cost_estimate_usd or 0.0 for _, weak, _ in pairs)
        all_strong_cost = sum(strong.cost_estimate_usd or 0.0 for _, _, strong in pairs)
        selected_tasks = {task_id for task_id, _, _, _ in selected}
        routed_cost = sum(
            (strong.cost_estimate_usd if task_id in selected_tasks else weak.cost_estimate_usd)
            or 0.0
            for task_id, weak, strong in pairs
        )
        cost_savings = None if routed_cost <= 0 else all_strong_cost / routed_cost
    else:
        all_weak_cost = all_strong_cost = routed_cost = cost_savings = None

    def pct(value: float) -> str:
        return f"{value * 100:.0f}%"

    lines = [
        "RouteLLM-style calibration",
        "--------------------------",
        f"Weak agent: {weak_agent}",
        f"Strong agent: {strong_agent}",
        f"Paired tasks: {len(pairs)}",
        f"Weak avg score: {weak_avg:.3f}",
        f"Strong avg score: {strong_avg:.3f}",
        f"Target PGR: {pct(target_pgr)}",
        f"Measured CPT({pct(target_pgr)}): {pct(strong_call_share)} strong-agent calls",
        f"Achieved PGR: {pct(achieved_pgr)}",
        f"Routed avg score: {routed_score:.3f}",
    ]
    if known_costs:
        lines.extend(
            [
                f"All-weak cost: {_money(all_weak_cost)}",
                f"All-strong cost: {_money(all_strong_cost)}",
                f"Routed cost: {_money(routed_cost)}",
                "Cost saving vs all-strong: "
                + ("n/a" if cost_savings is None else f"{cost_savings:.2f}x"),
            ]
        )
    lines.append("")
    lines.append("Strong-agent tasks:")
    for task_id, weak, strong, gain in selected:
        lines.append(
            f"- {task_id}: {weak.final_score:.2f} -> {strong.final_score:.2f} (gain {gain:.2f})"
        )
    return "\n".join(lines)


def cmd_calibrate(args: argparse.Namespace) -> int:
    reports = _load(args)
    if args.weak_agent or args.strong_agent:
        if not args.weak_agent or not args.strong_agent:
            print(
                "--weak-agent and --strong-agent must be provided together.",
                file=sys.stderr,
            )
            return 2
        print(
            _render_route_calibration(
                reports,
                weak_agent=args.weak_agent,
                strong_agent=args.strong_agent,
                target_pgr=args.target_pgr,
            )
        )
    else:
        print(_render_calibration(reports, min_score=args.min_score))
    return 0


def cmd_init_agents(args: argparse.Namespace) -> int:
    out = write_example_config(args.out)
    print(f"Wrote starter agent config to {out}")
    print("Edit the command arrays to match the terminal agents installed on your machine.")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    job_path = str(args.job)
    prompt = open(args.job, encoding="utf-8").read()
    if args.extra_prompt:
        prompt = prompt.rstrip() + "\n\n" + args.extra_prompt

    success_command = args.success_command if args.success_command else None
    out_paths = []
    for agent in load_agents(args.agents):
        print(f"Running {agent.name}...")
        out_paths.append(
            run_terminal_agent(
                agent,
                prompt,
                task_id=args.task_id or Path(args.job).stem,
                out_dir=args.out,
                job_path=job_path,
                workspace=args.workspace,
                reset_workspace=args.reset_workspaces,
                success_regex=args.success_regex,
                score_regex=args.score_regex,
                success_command=success_command,
            )
        )

    reports = rank_reports(analyze_many(load_conversations([str(args.out)])))
    print()
    print(render_table(reports))
    if reports:
        winner = reports[0]
        cost = "n/a" if winner.cost_estimate_usd is None else f"${winner.cost_estimate_usd:.4f}"
        print()
        print(
            "Recommendation: use "
            f"{winner.agent} for this job type "
            f"(efficiency {winner.convergence_efficiency:.3f}, "
            f"cost {cost}, "
            f"latency {winner.latency_estimate_ms / 1000:.1f}s)."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenomist",
        description="Compare how different AI agents solve the same task.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Compare agents in a table or JSON report.")
    _add_common(p_analyze)
    p_analyze.add_argument("--json", metavar="PATH", help="Write JSON report to PATH.")
    p_analyze.add_argument(
        "--with-trace",
        action="store_true",
        help="Include per-turn traces in the JSON report.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_trace = sub.add_parser("trace", help="Export per-turn traffic traces as CSV.")
    _add_common(p_trace)
    p_trace.add_argument("--csv", metavar="PATH", help="Write CSV to PATH (default: stdout).")
    p_trace.set_defaults(func=cmd_trace)

    p_formats = sub.add_parser("formats", help="List supported log formats.")
    p_formats.set_defaults(func=cmd_formats)

    p_calibrate = sub.add_parser(
        "calibrate",
        help="Summarize prior runs as a routing policy by success rate and cost per correct.",
    )
    _add_common(p_calibrate)
    p_calibrate.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="Minimum final_score to count as correct for calibration.",
    )
    p_calibrate.add_argument(
        "--weak-agent",
        default=None,
        help="Cheap/weak agent name for RouteLLM-style paired calibration.",
    )
    p_calibrate.add_argument(
        "--strong-agent",
        default=None,
        help="Expensive/strong agent name for RouteLLM-style paired calibration.",
    )
    p_calibrate.add_argument(
        "--target-pgr",
        type=float,
        default=0.8,
        help="Target performance gap recovered for paired calibration, e.g. 0.8.",
    )
    p_calibrate.set_defaults(func=cmd_calibrate)

    p_init = sub.add_parser(
        "init-agents",
        help="Write a starter config for terminal agents such as Codex, Gemini, Claude, or Cursor.",
    )
    p_init.add_argument("--out", default="agents.json", help="Where to write the starter config.")
    p_init.set_defaults(func=cmd_init_agents)

    p_route = sub.add_parser(
        "route",
        help="Run one job through several configured terminal agents and recommend a winner.",
    )
    p_route.add_argument("job", help="Text/Markdown file describing the job to give every agent.")
    p_route.add_argument(
        "--agents",
        required=True,
        help="JSON config from init-agents, edited with your local agent commands.",
    )
    p_route.add_argument("--out", default="runs/route", help="Directory for native run logs.")
    p_route.add_argument("--task-id", default=None, help="Task id for comparison grouping.")
    p_route.add_argument(
        "--workspace",
        default=None,
        help=(
            "Directory to copy once per agent before running. Use this for strong "
            "coding comparisons so each agent edits an isolated workspace."
        ),
    )
    p_route.add_argument(
        "--reset-workspaces",
        action="store_true",
        help="Replace existing per-agent workspace copies under OUT/workspaces.",
    )
    p_route.add_argument(
        "--extra-prompt",
        default=None,
        help="Additional instructions appended to the job before sending it to each agent.",
    )
    p_route.add_argument(
        "--success-regex",
        default=None,
        help="Regex that must match agent output for the run to count as correct.",
    )
    p_route.add_argument(
        "--score-regex",
        default=None,
        help=(
            "Regex with an optional first numeric capture group for final_score, "
            "e.g. 'score: ([0-9.]+)'."
        ),
    )
    p_route.add_argument(
        "--success-command",
        nargs=argparse.REMAINDER,
        default=None,
        help=(
            "Command run after each agent; exit code 0 marks the run correct, "
            "e.g. --success-command pytest -q. Put this option last."
        ),
    )
    p_route.set_defaults(func=cmd_route)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
