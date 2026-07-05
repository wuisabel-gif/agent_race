"""Smoke test for the capture harness (offline mock mode).

Runs harness/run_agent.py against the bundled example task with --mock (no API
key, no network), then feeds the emitted log back through Tokenomist to prove
the harness produces a valid, scorable native-format log.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "harness" / "run_agent.py"
TASK = REPO / "harness" / "tasks" / "merge_intervals"


def test_harness_mock_produces_scorable_log(tmp_path):
    out = tmp_path / "run.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "--task",
            str(TASK),
            "--out",
            str(out),
            "--agent",
            "MockAgent",
            "--model",
            "glm-5.1",
            "--task-id",
            "merge_intervals",
            "--mock",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr

    log = json.loads(out.read_text(encoding="utf-8"))
    # The mock fails once then fixes it: converges, with a retry and a correction.
    assert log["final_correct"] is True
    assert log["success_turn"] is not None
    assert any(t.get("is_retry") for t in log["turns"])
    assert any(t.get("is_correction") for t in log["turns"])

    # And the emitted log is readable + scorable by Tokenomist.
    from tokenomist import analyze, load_conversations

    convs = load_conversations([str(out)])
    assert len(convs) == 1
    report = analyze(convs[0])
    assert report.final_correct is True
    assert report.cost_estimate_usd is not None  # glm-5.1 is a known price-book id
    assert report.retry_loops >= 1
    assert report.correction_count >= 1
