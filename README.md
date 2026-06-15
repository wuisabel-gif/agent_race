# AgentTraceLab

**Compare how different AI agents solve the same task — by cost, speed, accuracy, and reasoning efficiency.**

AgentTraceLab converts multi-turn AI conversations (ChatGPT, Gemini, Claude, the
OpenAI Agents SDK, LangGraph, or your own custom agent) into **structured traffic
traces**, then analyzes latency, token cost, tool-call patterns, retries, and
convergence behavior. It turns a pile of raw conversation logs into an
apples-to-apples leaderboard of agent systems on the same task.

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
`agenttracelab analyze` produces.

## Why this exists

Agentic LLM workloads are bursty, multi-turn, and tool-heavy, which makes them
hard to reason about from raw logs. AgentTraceLab characterizes that workload:
it reconstructs the growing context each assistant turn re-reads (so input
tokens accumulate the way they really do against a stateless API), attributes
cost and latency per turn, and surfaces the retry loops and user corrections
that drive real-world inefficiency. The per-turn CSV trace is a ready-made input
for a queueing model or datacenter simulator.

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
pip install -e .                 # core library + CLI
pip install -e ".[dashboard]"    # adds the Streamlit dashboard
pip install -e ".[tokens]"       # adds tiktoken for exact token counts
pip install -e ".[dev]"          # adds pytest
```

The core library has **zero required dependencies**; `tiktoken`, `streamlit`,
and `pandas` are optional. Without `tiktoken`, tokens are estimated with a
deterministic ~4-chars/token heuristic.

## Usage

```bash
# Compare agents in a table
agenttracelab analyze data/samples

# Write a JSON report (add --with-trace to embed per-turn rows)
agenttracelab analyze data/samples --json reports.json

# Export the per-turn traffic trace as CSV
agenttracelab trace data/samples --csv traces.csv

# List supported log formats
agenttracelab formats
```

### Dashboard

```bash
streamlit run src/agenttracelab/dashboard.py
```

Upload logs or load the bundled samples to get the comparison table plus charts
for cost, latency, tokens-to-success, and convergence efficiency.

### Library

```python
from agenttracelab import load_conversations, analyze_many
from agenttracelab.report import rank_reports, render_table

reports = rank_reports(analyze_many(load_conversations(["data/samples"])))
print(render_table(reports))
```

## Supported formats

Formats are auto-detected from the JSON shape (override with `--format`):

| Format | Source | Shape |
| --- | --- | --- |
| `native` | AgentTraceLab schema | `{"turns": [...]}` with explicit metadata |
| `openai_chat` | ChatGPT / Chat Completions / Agents SDK | `{"messages": [{"role", "content"}]}` |
| `gemini` | Google `generateContent` | `{"contents": [{"role", "parts"}]}` |
| `langgraph` | LangChain / LangGraph state | `{"messages": [{"type", "content"}]}` |

See [`data/samples`](data/samples) for one example of each, all solving the same
`fix-parse-duration` task.

## Project layout

```
src/agenttracelab/
  models.py        normalized Conversation / Turn / ToolCall
  tokens.py        token estimation (tiktoken or heuristic)
  pricing.py       per-model price book + latency model
  parsers/         format detection + one parser per format
  analyzer.py      trace construction + aggregate metrics
  report.py        table / JSON / CSV rendering and ranking
  cli.py           command-line interface
  dashboard.py     Streamlit dashboard (optional)
data/samples/      example logs, one per format
tests/             pytest suite
```

## Tests

```bash
pytest
```

## License

MIT
