"""End-to-end tests for the command-line interface.

These drive ``cli.main`` the way a user would, asserting on stdout and on any
files written, so the wiring from parsing → analysis → rendering stays covered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenomist.cli import main

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_analyze_table_to_stdout(capsys):
    rc = main(["analyze", str(SAMPLES)])
    out = capsys.readouterr().out
    assert rc == 0
    # Header and at least one known agent row rendered.
    assert "Agent" in out and "Cost" in out
    assert "ChatGPT" in out


def test_analyze_json_report_is_valid(tmp_path, capsys):
    out_path = tmp_path / "reports.json"
    rc = main(["analyze", str(SAMPLES), "--json", str(out_path)])
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload
    # Every report carries the fields the table/consumers rely on.
    for rep in payload:
        assert "agent" in rep
        assert "cost_estimate_usd" in rep
        assert "convergence_efficiency" in rep


def test_trace_csv_has_header_and_rows(tmp_path, capsys):
    out_path = tmp_path / "trace.csv"
    rc = main(["trace", str(SAMPLES), "--csv", str(out_path)])
    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("task_id,agent,model")
    assert len(lines) > 1


def test_formats_lists_parsers(capsys):
    rc = main(["formats"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "native" in out
    assert "openai_chat" in out


def test_calibrate_reports_cost_per_correct(capsys):
    rc = main(["calibrate", str(SAMPLES)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cost/correct" in out
    assert "Calibration recommendation" in out


def test_calibrate_reports_route_llm_style_cpt(tmp_path, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    for task_id, weak_score, strong_score in [
        ("easy", 1.0, 1.0),
        ("medium", 0.5, 1.0),
        ("hard", 0.0, 1.0),
    ]:
        for agent, model, score in [
            ("CheapAgent", "gpt-4o-mini", weak_score),
            ("StrongAgent", "gpt-4o", strong_score),
        ]:
            payload = {
                "agent": agent,
                "model": model,
                "task_id": task_id,
                "success_turn": 1 if score >= 1.0 else None,
                "final_correct": score >= 1.0,
                "final_score": score,
                "turns": [
                    {"role": "user", "content": f"solve {task_id}"},
                    {"role": "assistant", "content": "answer", "latency_ms": 10},
                ],
            }
            (logs / f"{agent}-{task_id}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    rc = main(
        [
            "calibrate",
            str(logs),
            "--weak-agent",
            "CheapAgent",
            "--strong-agent",
            "StrongAgent",
            "--target-pgr",
            "0.8",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "RouteLLM-style calibration" in out
    assert "Measured CPT(80%)" in out
    assert "Strong-agent tasks:" in out


def test_unknown_model_reports_na(tmp_path, capsys):
    # A log naming a model absent from the price book must render cost as n/a.
    src = json.loads((SAMPLES / "claude_native.json").read_text(encoding="utf-8"))
    src["model"] = "frobnicate-9000-ultra"
    src["agent"] = "MysteryAgent"
    log = tmp_path / "unknown.json"
    log.write_text(json.dumps(src), encoding="utf-8")

    rc = main(["analyze", str(log)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "n/a" in out


def test_custom_prices_flag_changes_cost(tmp_path, capsys):
    prices = {
        "default_output_tokens_per_sec": 60,
        "models": [{"family": "gpt-4o", "input": 999.0, "output": 999.0, "tps": 80}],
    }
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(prices), encoding="utf-8")

    chatgpt = SAMPLES / "chatgpt_openai.json"
    rc = main(["analyze", str(chatgpt), "--prices", str(price_path)])
    out = capsys.readouterr().out
    assert rc == 0
    # An absurd override price should push the cost well above a dollar.
    assert "$0.0" not in out


def test_init_agents_writes_config(tmp_path, capsys):
    out = tmp_path / "agents.json"
    rc = main(["init-agents", "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "starter agent config" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "agents" in payload
    assert any(agent["name"] == "Codex" for agent in payload["agents"])


def test_route_runs_terminal_agents_and_recommends(tmp_path, capsys):
    job = tmp_path / "job.md"
    job.write_text("Say DONE if you can solve this.", encoding="utf-8")
    agents = tmp_path / "agents.json"
    agents.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "name": "FastAgent",
                        "model": "gpt-4o-mini",
                        "command": [sys.executable, "-c", "print('DONE')"],
                    },
                    {
                        "name": "WrongAgent",
                        "model": "gpt-4o-mini",
                        "command": [sys.executable, "-c", "print('not solved')"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "runs"

    rc = main(
        [
            "route",
            str(job),
            "--agents",
            str(agents),
            "--out",
            str(out_dir),
            "--success-regex",
            "DONE",
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0
    assert "Recommendation: use FastAgent" in printed
    assert (out_dir / "FastAgent.json").exists()
    assert (out_dir / "WrongAgent.json").exists()


def test_route_success_command_runs_in_isolated_workspaces(tmp_path, capsys):
    job = tmp_path / "job.md"
    job.write_text("Fix solution.py so pytest passes.", encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
    (workspace / "test_solution.py").write_text(
        "from solution import answer\n\n\ndef test_answer():\n    assert answer() == 42\n",
        encoding="utf-8",
    )

    fixer = (
        "from pathlib import Path; "
        "Path('solution.py').write_text('def answer():\\n    return 42\\n')"
    )
    agents = tmp_path / "agents.json"
    agents.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "name": "Fixer",
                        "model": "gpt-4o-mini",
                        "command": [sys.executable, "-c", fixer],
                    },
                    {
                        "name": "Observer",
                        "model": "gpt-4o-mini",
                        "command": [sys.executable, "-c", "print('no change')"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "runs"

    rc = main(
        [
            "route",
            str(job),
            "--agents",
            str(agents),
            "--out",
            str(out_dir),
            "--workspace",
            str(workspace),
            "--reset-workspaces",
            "--success-command",
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0
    assert "Recommendation: use Fixer" in printed
    assert "return 0" in (workspace / "solution.py").read_text(encoding="utf-8")
    assert "return 42" in (out_dir / "workspaces" / "Fixer" / "solution.py").read_text(
        encoding="utf-8"
    )
    assert "return 0" in (out_dir / "workspaces" / "Observer" / "solution.py").read_text(
        encoding="utf-8"
    )
