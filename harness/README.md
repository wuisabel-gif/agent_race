# Capture harness

`run_agent.py` runs one coding task against one model, with a test-feedback
loop, and writes a **native-format** Tokenomist log with **real token usage
and real wall-clock latency** baked in. Point it at any OpenAI-compatible
endpoint, so a single script covers whatever API key you have.

Its output feeds straight into the analyzer:

```bash
tokenomist analyze runs/         # a directory of logs from several agents
```

## Install

```bash
pip install -e ".[capture]"   # adds the openai client (used for live runs)
```

`--mock` needs no dependencies and no network — use it to try the pipeline.

## Quick start (offline, no API key)

```bash
python harness/run_agent.py \
  --task harness/tasks/merge_intervals \
  --out runs/mock.json --agent "Demo" --model glm-5.1 --mock
tokenomist analyze runs
```

The mock model fails once, gets the test failure fed back, then fixes it — so
you can see a retry, a correction, and a converged run without spending a token.

## A task is a directory

```
tasks/<task_id>/
  task.md            # the spec shown to the model
  solution.py        # starter file the model rewrites (the file under test)
  test_solution.py   # ground-truth pytest — NEVER shown to the model
  reference_solution.py  # OPTIONAL: used only by --mock to produce a passing file
```

Rules for good tasks:

- The tests are the ground truth. Keep them out of the model's view (the harness
  copies only `solution.py` + `test_solution.py` into an isolated temp dir, and
  only shows the model `task.md` + the current `solution.py`).
- Make `task.md` specify the exact function name/signature so a pass is
  unambiguous. Cover edge cases in the tests (empty input, boundaries, "does not
  mutate input", etc.) — that's what separates a real fix from a lucky one.
- **Freeze your tasks before you see which model wins.** Picking tasks after the
  fact is how comparisons lie.

Copy `tasks/merge_intervals` as a template for your own three tasks.

## Live runs — endpoint presets

Set the matching API-key env var, then pass an endpoint preset. `--api-model` is
the string sent to the API; `--model` is the **price-book id** written to the log
(keep it a known family from `prices.json` so cost isn't `n/a`).

```bash
# Z.ai / GLM  (OpenAI-compatible endpoint)
export ZAI_API_KEY=...
python harness/run_agent.py --task harness/tasks/merge_intervals --out runs/glm51.json \
  --agent "GLM-5.1" --provider zai \
  --base-url https://api.z.ai/api/paas/v4 --api-model glm-4.6 --model glm-5.1 \
  --api-key-env ZAI_API_KEY --i-understand-code-execution

# DeepSeek
export DEEPSEEK_API_KEY=...
python harness/run_agent.py --task harness/tasks/merge_intervals --out runs/deepseek.json \
  --agent "DeepSeek V4" --provider deepseek \
  --base-url https://api.deepseek.com --api-model deepseek-chat --model deepseek-v4-pro \
  --api-key-env DEEPSEEK_API_KEY --i-understand-code-execution

# OpenAI (omit --base-url)
export OPENAI_API_KEY=...
python harness/run_agent.py --task harness/tasks/merge_intervals --out runs/gpt.json \
  --agent "GPT-5.4" --provider openai \
  --api-model gpt-5.4 --model gpt-5.4 --api-key-env OPENAI_API_KEY \
  --i-understand-code-execution
```

> Endpoint URLs and API model strings change; check your provider's docs. The
> `--api-model` you send is the provider's own id, which may differ from the
> price-book `--model`.

## The four-agent lineup from one API key

You don't need four providers. Vary tier and scaffold, keep the **same
`--task-id`**, write into one directory, then analyze:

```bash
D=runs/merge_intervals; mkdir -p $D
# 1) frontier tier, scaffolded (test-feedback loop)
run ... --agent "GLM-5.1"       --model glm-5.1 --max-turns 6 --out $D/glm51.json
# 2) cheaper tier, scaffolded
run ... --agent "GLM-5"         --model glm-5   --max-turns 6 --out $D/glm5.json
# 3) a second provider/model if you have one, scaffolded
run ... --agent "DeepSeek V4"   --model deepseek-v4-pro --max-turns 6 --out $D/ds.json
# 4) SCAFFOLD CONTRAST: same frontier model, raw single-shot (no feedback)
run ... --agent "GLM-5.1 (raw)" --model glm-5.1 --max-turns 1 --out $D/glm51_raw.json

tokenomist analyze runs/merge_intervals
```

`--max-turns 1` is the raw baseline (one shot, no test feedback); `--max-turns
6` is the scaffolded agent. Raw-vs-scaffolded on identical weights is the most
interesting axis this tool can show. Run each config across all three of your
tasks (same `--task-id` per task) and pool the logs.

Because models are nondeterministic, run each config **a few times** and keep all
the logs — one run is an anecdote. Report the spread in your write-up.

## Security

`run_agent.py` executes model-generated Python via `pytest`. Only run tasks you
trust, ideally inside a container or throwaway VM. Live runs require the explicit
`--i-understand-code-execution` flag; `--mock` never executes model output beyond
the canned reference solution you ship with the task.

## What the harness measures (all real, not estimated)

| Field in the log | Source |
| --- | --- |
| `input_tokens` / `output_tokens` | the API response `usage` |
| `latency_ms` | wall-clock around each API call |
| `success_turn` | first assistant turn after which all tests pass |
| `is_retry` | assistant turns after the first |
| `is_correction` | the harness's test-failure feedback turns |
| `final_score` | fraction of tests passing at the end (`1.0` ⇒ `final_correct`) |
| `tool_calls[].ok` | whether the patch parsed and whether the tests passed |

Cost is then computed by Tokenomist from `model` + real token counts, so your
numbers are exact rather than price-book estimates.
