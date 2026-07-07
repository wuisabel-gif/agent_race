# Tokenomist

[![CI](https://github.com/wuisabel-gif/Tokenomist/actions/workflows/ci.yml/badge.svg)](https://github.com/wuisabel-gif/Tokenomist/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The token economist: measure what your AI agents actually spend — and what
actually works — then pick the right agent for the job.**

Agents differ wildly in what they cost, how fast they converge, and how often
they're *right*. Tokenomist turns that into numbers: run several agents on the
same task (or import logs you already have), and get an apples-to-apples
leaderboard of correctness, cost, latency, retries, and efficiency — including
the number most comparisons skip, **cost per correct solution**.

📊 **[Try the live browser demo →](https://wuisabel-gif.github.io/Tokenomist/)** &nbsp;·&nbsp; no install, runs entirely in your browser.

```bash
pip install -e .
tokenomist analyze data/samples
```

Three ways to use it:

1. **Analyze logs you already have.** Drop ChatGPT, Gemini, Claude, OpenAI
   Agents SDK, LangGraph, or native JSON exports into a folder and run
   `tokenomist analyze <folder>`. Tokenomist reconstructs each run as a
   structured traffic trace and ranks the agents.
2. **Route one job through terminal agents.** Configure commands for Codex,
   Gemini, Claude, Zhipu/GLM, DeepSeek, Cursor, or any local agent CLI, then run
   `tokenomist route job.md --agents agents.json`. Every agent gets the same
   job; you get a recommendation backed by measurements.
3. **Generate measured benchmark logs.** Use `harness/run_agent.py` on a coding
   task with hidden pytest tests, then analyze the logs it writes. This is the
   path for reporting cost per *correct* solution.

```
Agent            Model          Turns  →Success  In tok  Out tok  Tools  Tool OK  Retries  Fixes  Latency  Cost     Score  Efficiency
---------------  -------------  -----  --------  ------  -------  -----  -------  -------  -----  -------  -------  -----  ----------
Custom Agent     claude-haiku   3      1         ...     ...      2      100%     0        0      ...      $...     0.95   0.901
Claude           claude-sonnet  5      4         ...     ...      3      100%     0        0      ...      $...     1.00   0.612
ChatGPT          gpt-4o         4      2         ...     ...      1      100%     0        0      ...      $...     1.00   0.588
LangGraph Agent  gpt-4o-mini    7      4         ...     ...      3      100%     0        0      ...      $...     0.90   0.501
Gemini           gemini-1.5-pro 11     -         ...     ...      3      100%     3        4      ...      $...     0.60   0.000
```

## Browser demo

[`index.html`](index.html) is a **self-contained, zero-dependency demo** — the
token/cost/latency models are ported to JavaScript and run entirely in the
browser. Just open the file (double-click, or `open index.html`); nothing is
uploaded anywhere. It ships with the same five sample logs, auto-detects
formats, and supports drag-and-drop of your own conversation JSON.

It ships with **five short example workloads**, each won by a different agent, so
no single system looks artificially dominant — pick one from the dropdown:

| Example | Winner | Why |
| --- | --- | --- |
| Fix a failing unit test | Claude | clean one-pass fix; others need a correction |
| Summarize a 40-page contract | Gemini | long-context, one shot, pennies |
| Web research with a citation | ChatGPT | searches and cites instead of guessing |
| Quick factual question | Custom Agent | everyone's right, so cheapest wins |
| Debug a red CI build | LangGraph | finds the real root cause, not a workaround |

It renders:

- an animated **race** where each agent is a car whose speed is its convergence
  efficiency — the most efficient agent takes the checkered flag (🥇🥈🥉) and
  runs that never converged stall before the finish line,
- a ranked comparison table (click any column to re-sort),
- summary cards (most efficient / cheapest / fastest agent),
- bar charts for cost, latency, convergence efficiency, and tokens-to-success, and
- a per-turn traffic-trace chart showing cumulative cost as the context window
  grows, with retries and user corrections marked inline.

Mirrors the Python CLI output, so it doubles as a quick visual of what
`tokenomist analyze` produces.

## Why this exists

Agentic LLM workloads are bursty, multi-turn, and tool-heavy, which makes them
hard to reason about from raw logs — and expensive to get wrong. Tokenomist
does the accounting: it reconstructs the growing context each assistant turn
re-reads (so input tokens accumulate the way they really do against a stateless
API), attributes cost and latency per turn, and surfaces the retry loops and
user corrections that drive real-world inefficiency. The per-turn CSV trace is
a ready-made input for a queueing model or datacenter simulator.

## Metrics

| Metric | Meaning |
| --- | --- |
| `turns_to_success` | Assistant turns before a useful answer |
| `input` / `output_tokens` | Prompt vs. completion token volume |
| `tool_calls`, `tool_success_rate` | Tool usage and how often it worked |
| `retry_loops` | Assistant turns that retried after a failure |
| `correction_count` | Times the user had to redirect the agent |
| `latency_estimate_ms` | Throughput-modeled generation latency |
| `cost_estimate_usd` | Token cost using a configurable price book |
| `convergence_efficiency` | 0–1: correct answers reached with the least budget and fewest corrections |

## Install

```bash
pip install -e .              # core library + CLI
pip install -e ".[dashboard]" # adds the Streamlit dashboard
pip install -e ".[tokens]"    # adds tiktoken for exact token counts
```

The core library has **zero required dependencies**; `tiktoken`, `streamlit`,
and `pandas` are optional. Without `tiktoken`, tokens are estimated with a
deterministic ~4-chars/token heuristic.

### From source (for development)

```bash
git clone https://github.com/wuisabel-gif/Tokenomist
cd Tokenomist
pip install -e ".[dev]"   # editable install with pytest + ruff
pytest                    # run the test suite
```

### Build the package

```bash
python -m build
python -m twine check dist/*
pipx run --spec dist/tokenomist-0.1.0-py3-none-any.whl tokenomist formats
```

The package installs the `tokenomist` console command and includes the bundled
price book used for cost and latency estimates.

## Usage

```bash
# Compare agents in a table
tokenomist analyze data/samples

# Turn prior runs into a measured routing recommendation
tokenomist calibrate runs/fix-tests

# Compute a RouteLLM-style strong/weak routing threshold from paired runs
tokenomist calibrate runs/fix-tests \
  --weak-agent "DeepSeek" \
  --strong-agent "Claude Code" \
  --target-pgr 0.8

# Write a JSON report (add --with-trace to embed per-turn rows)
tokenomist analyze data/samples --json reports.json

# Export the per-turn traffic trace as CSV
tokenomist trace data/samples --csv traces.csv

# Use your own up-to-date price book instead of the bundled one
tokenomist analyze data/samples --prices my_prices.json

# Create a terminal-agent config, then run the same job through every agent
tokenomist init-agents --out agents.json
tokenomist route job.md --agents agents.json --out runs/job1

# List supported log formats
tokenomist formats
```

### Pricing

Model prices live in [`src/tokenomist/prices.json`](src/tokenomist/prices.json),
not in code — each entry is a model *family* (a stable name prefix like
`claude-sonnet-4-6`) plus input/output rates per million tokens, an optional
cached-input rate, and a rough throughput for latency estimation. Lookup matches
by longest family prefix, so dated model ids such as
`claude-sonnet-4-6-20250514` still resolve. An **unknown model reports cost as
`n/a`** rather than a fabricated number. Prices are approximate public list
prices for *relative* comparison, verified `2026-07-04`; update the file (and
bump `last_verified`) or pass `--prices` to keep them current.

### Dashboard

```bash
streamlit run src/tokenomist/dashboard.py
```

Upload logs or load the bundled samples to get the comparison table plus charts
for cost, latency, tokens-to-success, and convergence efficiency.

### Library

```python
from tokenomist import load_conversations, analyze_many
from tokenomist.report import rank_reports, render_table

reports = rank_reports(analyze_many(load_conversations(["data/samples"])))
print(render_table(reports))
```

## Route jobs through terminal agents

This is the "which agent should I use?" workflow. You describe a job once, map
each installed agent to a terminal command, and Tokenomist runs them all,
captures native logs, analyzes the results, and prints a recommendation.

The design borrows the practical lesson from model-routing systems like
RouteLLM: do not send every request to the most expensive model by default.
Instead, measure your own workload, find which cheaper agents are reliable for
which task shapes, and reserve stronger agents for the cases where they are
worth the extra cost. Tokenomist does that with real run logs and success checks
rather than a pretrained router.

```bash
tokenomist init-agents --out agents.json
```

Edit `agents.json` so each command matches the tools installed on your machine:

```json
{
  "agents": [
    {
      "name": "Claude Code",
      "provider": "anthropic",
      "model": "claude-sonnet-5",
      "command": ["claude", "-p", "{prompt}"]
    },
    {
      "name": "Codex",
      "provider": "openai",
      "model": "gpt-5.4",
      "command": ["codex", "exec", "{prompt}"]
    },
    {
      "name": "Gemini CLI",
      "provider": "google",
      "model": "gemini-3.1-pro",
      "command": ["gemini", "-p", "{prompt}"]
    },
    {
      "name": "Zhipu GLM",
      "provider": "zai",
      "model": "glm-5.1",
      "command": ["zai", "chat", "--model", "glm-5.1", "{prompt}"]
    },
    {
      "name": "DeepSeek",
      "provider": "deepseek",
      "model": "deepseek-v4-pro",
      "command": ["deepseek", "chat", "--model", "deepseek-chat", "{prompt}"]
    },
    {
      "name": "Cursor Agent",
      "provider": "cursor",
      "model": "cursor-agent",
      "command": ["cursor-agent", "--print", "{prompt}"]
    }
  ]
}
```

The command names above are examples. If you call Zhipu or DeepSeek through a
different wrapper, local script, OpenAI-compatible proxy, or company CLI, keep
the `name` / `model` fields and change only the `command` array.

Then write a job:

```markdown
# Job

Fix the failing tests in this repo. Explain what changed and stop when tests pass.
```

Run every configured agent on that same job:

```bash
tokenomist route job.md --agents agents.json --out runs/fix-tests \
  --success-regex "tests pass|DONE|fixed"
```

For coding work, use an objective success command when possible:

```bash
tokenomist route job.md --agents agents.json --out runs/fix-tests \
  --success-command pytest -q
```

For the strongest comparison, give each agent its own copy of the repo and run
the check inside that copy:

```bash
tokenomist route job.md --agents agents.json --out runs/fix-tests \
  --workspace . --reset-workspaces \
  --success-command pytest -q
```

This prevents one agent's edits from affecting another agent's run. The original
repo stays untouched; each agent works under `runs/fix-tests/workspaces/<agent>/`.

After a few runs, treat the leaderboard as your calibration set: if a cheaper
agent repeatedly passes the strong check for a task family, route that work to
it; if it fails or needs too many retries, raise the quality threshold and use a
stronger agent. The headline number to watch is cost per correct solution.

```bash
tokenomist calibrate runs/fix-tests
```

Tokenomist also includes a direct adaptation of the RouteLLM paper's routing
metrics. If you have paired runs where a cheap/weak agent and an expensive/strong
agent both attempted the same `task_id` values, ask how often the strong agent
would need to be called to recover a target share of the quality gap:

```bash
tokenomist calibrate runs/fix-tests \
  --weak-agent "DeepSeek" \
  --strong-agent "Claude Code" \
  --target-pgr 0.8
```

That prints the measured `CPT(80%)`: the smallest share of calls sent to the
strong agent that recovers about 80% of the score gap between the weak-only and
strong-only baselines. In paper terms, this is a calibration-stage oracle rather
than a trained router: it tells you whether a task family has enough cheap-agent
headroom to justify building or using a live router.

That makes Tokenomist a terminal decision layer: compare Codex vs Gemini vs
Claude vs Zhipu/GLM vs DeepSeek vs Cursor on your actual task shape, then choose
the agent with the best mix of correctness, cost, latency, and retry behavior.
The command adapter is generic on purpose — if a tool can be called
non-interactively from your terminal, it can be plugged in.

## Research reference

The strong/weak routing calibration is inspired by:

> Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu,
> Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica.
> **RouteLLM: Learning to Route LLMs with Preference Data.**
> ICLR 2025. arXiv:2406.18665.

Tokenomist does not train a RouteLLM router. It adapts the paper's practical
cost-quality framing — weak model, strong model, performance gap recovered, and
call-performance threshold — to terminal-agent logs with objective success
checks.

## Capture your own agent runs

The [`harness/`](harness) directory has a provider-agnostic runner that solves a
coding task against any OpenAI-compatible model (Z.ai/GLM, DeepSeek, OpenAI, …)
with a test-feedback loop, capturing **real token usage and latency** into a
native-format log:

```bash
pip install -e ".[capture]"
python harness/run_agent.py --task harness/tasks/merge_intervals \
  --out runs/demo.json --model glm-5.1 --mock   # offline demo, no API key
tokenomist analyze runs
```

Because each task ships a hidden pytest suite as ground truth, you get objective
**correctness** alongside the efficiency metrics — enough to rank agents by
*cost per correct solution*. See [`harness/README.md`](harness/README.md) for
endpoint presets and the four-agent recipe, and
[`docs/case-study.md`](docs/case-study.md) for a write-up scaffold.

## Supported formats

Formats are auto-detected from the JSON shape (override with `--format`):

| Format | Source | Shape |
| --- | --- | --- |
| `native` | Tokenomist schema | `{"turns": [...]}` with explicit metadata |
| `openai_chat` | ChatGPT / Chat Completions / Agents SDK | `{"messages": [{"role", "content"}]}` |
| `gemini` | Google `generateContent` | `{"contents": [{"role", "parts"}]}` |
| `langgraph` | LangChain / LangGraph state | `{"messages": [{"type", "content"}]}` |

See [`data/samples`](data/samples) for one example of each. **For the exact
field-by-field upload contract, see [`FORMAT.md`](FORMAT.md)** (also available
in the browser demo under "How should my logs look?").

## Project layout

```
src/tokenomist/
  models.py        normalized Conversation / Turn / ToolCall
  tokens.py        token estimation (tiktoken or heuristic)
  pricing.py       price-book loader + latency model (prefix/family matching)
  prices.json      editable price book (per-model rates, verified 2026-07-04)
  parsers/         format detection + one parser per format
  analyzer.py      trace construction + aggregate metrics
  report.py        table / JSON / CSV rendering and ranking
  cli.py           command-line interface
  dashboard.py     Streamlit dashboard (optional)
data/samples/      example logs, one per format
harness/           capture runner + example pytest-backed task
docs/              case-study scaffold for reporting results
tests/             pytest suite
```

## Tests

```bash
pytest
```

## License

MIT
