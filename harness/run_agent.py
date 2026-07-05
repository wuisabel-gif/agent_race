#!/usr/bin/env python3
"""Tokenomist capture harness.

Run one coding task against one model, with a test-feedback loop, and emit a
**native-format** Tokenomist log (real token usage + real wall-clock latency
baked in). Point it at any OpenAI-compatible endpoint — Z.ai (GLM), DeepSeek,
OpenAI, or a local server — so a single script covers whatever key you have.

A "task" is a directory:

    tasks/<task_id>/
      task.md            # the natural-language spec shown to the model
      solution.py        # starter file the model rewrites (the file under test)
      test_solution.py   # ground-truth pytest (NEVER shown to the model)

The loop each turn: show the model the task + current solution -> it returns the
complete updated file -> we write it and run pytest -> if it fails we feed the
failures back as a correction and retry, up to --max-turns.

Metrics that fall out, all measured (not guessed):
  * success_turn   = first assistant turn after which every test passes
  * is_retry       = assistant turns after the first
  * is_correction  = the harness's test-failure feedback turns
  * final_score    = fraction of tests passing at the end (1.0 => final_correct)
  * input/output_tokens, latency_ms = from the API response + a wall-clock timer

The 4-agent comparison comes from running this several times with the SAME
--task-id into one output dir, then: `tokenomist analyze <dir>`. Vary --model
(tier) and --max-turns (scaffold: 1 = raw single-shot, >1 = test-feedback loop).

SECURITY: this executes model-generated code via pytest in a temp dir. Only run
tasks you trust, ideally in a container/VM. Pass --i-understand-code-execution.

Requires the openai client for live runs:  pip install "tokenomist[capture]"
Use --mock to try the whole pipeline with no API key and no network.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a coding agent. You are given a task and the current contents of a "
    "single Python file. Reply with the COMPLETE updated contents of that file "
    "and nothing else, inside one ```python code fence. No prose outside the fence."
)

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_PYTEST_PASS = re.compile(r"(\d+) passed")
_PYTEST_FAIL = re.compile(r"(\d+) failed")
_PYTEST_ERROR = re.compile(r"(\d+) error")


def _extract_code(reply: str) -> str | None:
    """Return the last fenced code block in a model reply, or None."""
    blocks = _CODE_FENCE.findall(reply)
    return blocks[-1].strip() + "\n" if blocks else None


def _run_pytest(workdir: Path, test_file: str) -> tuple[bool, int, int, str]:
    """Run pytest in ``workdir``. Return (all_passed, n_passed, n_total, output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test_file],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (proc.stdout + "\n" + proc.stderr).strip()
    passed = int(m.group(1)) if (m := _PYTEST_PASS.search(out)) else 0
    failed = int(m.group(1)) if (m := _PYTEST_FAIL.search(out)) else 0
    errored = int(m.group(1)) if (m := _PYTEST_ERROR.search(out)) else 0
    total = passed + failed + errored
    all_passed = proc.returncode == 0 and total > 0 and failed == 0 and errored == 0
    return all_passed, passed, max(total, 1), out


def _truncate(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...(truncated)"


class _MockClient:
    """Offline stand-in: returns a wrong patch first, then a correct one.

    Lets you exercise the full harness (retry + correction + success) with no
    API key. It reads the ground-truth test to synthesize a passing file, so the
    example task "just works" as a demo.
    """

    def __init__(self, task_dir: Path):
        self._task_dir = task_dir
        self._call = 0

    def complete(self, messages):
        self._call += 1
        time.sleep(0.05)
        if self._call == 1:
            reply = "```python\ndef solve(*args, **kwargs):\n    return None  # first attempt\n```"
        else:
            # Second attempt: use the reference solution if the task ships one.
            ref = self._task_dir / "reference_solution.py"
            body = ref.read_text() if ref.exists() else "def solve(*a, **k):\n    return None\n"
            reply = f"```python\n{body}```"
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        completion_tokens = max(1, len(reply) // 4)
        return reply, prompt_tokens, completion_tokens


class _OpenAICompatClient:
    """Thin wrapper over the openai SDK pointed at any compatible base_url."""

    def __init__(self, base_url: str | None, api_key_env: str, api_model: str, temperature: float):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "The openai client is required for live runs. Install it with:\n"
                '    pip install "tokenomist[capture]"\n'
                "or run with --mock to try the pipeline offline."
            ) from exc
        import os

        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"No API key found in ${api_key_env}. Set it, or use --mock.")
        self._client = OpenAI(base_url=base_url, api_key=key)
        self._model = api_model
        self._temperature = temperature

    def complete(self, messages):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        reply = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return reply, prompt_tokens, completion_tokens


def run(args: argparse.Namespace) -> int:
    task_dir = Path(args.task).resolve()
    spec = (task_dir / "task.md").read_text(encoding="utf-8")
    solution_name = args.solution_file
    starter = (task_dir / solution_name).read_text(encoding="utf-8")
    test_name = args.test_file

    if not args.mock and not args.i_understand_code_execution:
        raise SystemExit(
            "This runs model-generated code via pytest. Re-run with "
            "--i-understand-code-execution (ideally inside a container/VM)."
        )

    client = (
        _MockClient(task_dir)
        if args.mock
        else _OpenAICompatClient(args.base_url, args.api_key_env, args.api_model, args.temperature)
    )

    # Isolated working copy so the model can't see or clobber the ground truth.
    workdir = Path(tempfile.mkdtemp(prefix="atl_task_"))
    shutil.copy(task_dir / solution_name, workdir / solution_name)
    shutil.copy(task_dir / test_name, workdir / test_name)

    turns: list[dict] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    current = starter
    success_turn: int | None = None
    last_fraction = 0.0

    # Turn 0: the task prompt (a user turn).
    user_prompt = f"{spec}\n\nCurrent `{solution_name}`:\n```python\n{current}```"
    turns.append({"role": "user", "content": user_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(1, args.max_turns + 1):
        t0 = time.perf_counter()
        reply, in_tok, out_tok = client.complete(messages)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        messages.append({"role": "assistant", "content": reply})

        code = _extract_code(reply)
        patch_ok = code is not None
        if patch_ok:
            current = code
            (workdir / solution_name).write_text(current, encoding="utf-8")
            all_passed, n_pass, n_total, test_out = _run_pytest(workdir, test_name)
            last_fraction = n_pass / n_total
            tool_calls = [
                {"name": "apply_patch", "arguments": {"path": solution_name}, "ok": True},
                {
                    "name": "run_tests",
                    "arguments": {"path": test_name},
                    "ok": all_passed,
                    "result": f"{n_pass}/{n_total} passed",
                },
            ]
        else:
            all_passed = False
            test_out = "Model reply contained no ```python code fence."
            tool_calls = [
                {"name": "apply_patch", "arguments": {"path": solution_name}, "ok": False}
            ]

        assistant_index = len(turns)
        turns.append(
            {
                "role": "assistant",
                "content": reply,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": latency_ms,
                "tool_calls": tool_calls,
                "is_retry": attempt > 1,
            }
        )

        if all_passed:
            success_turn = assistant_index
            break

        if attempt < args.max_turns:
            feedback = (
                "The tests did not all pass:\n```\n"
                + _truncate(test_out)
                + "\n```\nReturn the COMPLETE corrected file in one ```python fence."
            )
            turns.append({"role": "user", "content": feedback, "is_correction": True})
            messages.append({"role": "user", "content": feedback})

    shutil.rmtree(workdir, ignore_errors=True)

    final_correct = success_turn is not None
    final_score = 1.0 if final_correct else round(last_fraction, 3)
    log = {
        "agent": args.agent,
        "model": args.model or args.api_model,
        "provider": args.provider,
        "task_id": args.task_id or task_dir.name,
        "success_turn": success_turn,
        "final_correct": final_correct,
        "final_score": final_score,
        "turns": turns,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    status = "PASS" if final_correct else f"partial ({final_score})"
    a_turns = sum(1 for t in turns if t["role"] == "assistant")
    print(
        f"[{args.agent}] task={log['task_id']} model={log['model']} "
        f"-> {status} in {a_turns} assistant turn(s); wrote {out_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_agent",
        description="Capture an agent solving a coding task into an Tokenomist log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Endpoint presets (set the matching *_API_KEY env var):\n"
            "  Z.ai / GLM   --base-url https://api.z.ai/api/paas/v4 "
            "--api-model glm-4.6 --model glm-5.1 --api-key-env ZAI_API_KEY\n"
            "  DeepSeek     --base-url https://api.deepseek.com "
            "--api-model deepseek-chat --model deepseek-v4-pro --api-key-env DEEPSEEK_API_KEY\n"
            "  OpenAI       (omit --base-url) --api-model gpt-4o --model gpt-4o "
            "--api-key-env OPENAI_API_KEY\n\n"
            "  --api-model is the string sent to the API; --model is the price-book\n"
            "  id written to the log (used by Tokenomist for cost). Keep --model a\n"
            "  known family from prices.json so cost isn't n/a.\n"
        ),
    )
    p.add_argument("--task", required=True, help="Path to a task directory (see harness/tasks/).")
    p.add_argument("--out", required=True, help="Where to write the JSON log.")
    p.add_argument("--agent", default="Agent", help="Display name for this run.")
    p.add_argument(
        "--task-id",
        default=None,
        help="Overrides the task dir name; keep it the SAME across the agents you want compared.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Price-book model id for cost (e.g. glm-5.1). Defaults to --api-model.",
    )
    p.add_argument("--provider", default=None, help="Free-form provider label for the log.")
    p.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Turn cap; 1 = raw single-shot (no test feedback), >1 = scaffolded retry loop.",
    )
    p.add_argument(
        "--solution-file", default="solution.py", help="File under test the model edits."
    )
    p.add_argument("--test-file", default="test_solution.py", help="Ground-truth pytest file.")
    # Live-endpoint options.
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (omit for OpenAI).")
    p.add_argument("--api-model", default=None, help="Model string sent to the API.")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var holding the API key.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--mock", action="store_true", help="Offline: fake the model (no key/network).")
    p.add_argument(
        "--i-understand-code-execution",
        action="store_true",
        help="Required for live runs: acknowledges running model code via pytest.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.mock and not args.api_model:
        raise SystemExit("Provide --api-model for a live run (or use --mock).")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
