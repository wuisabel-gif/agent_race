# Case study: comparing agents on <TASK DOMAIN>

<!--
This scaffold turns a set of Tokenomist runs into a credible write-up. Fill
the blanks; delete these comments as you go. The sections in **bold** below are
the ones that separate a credible study from marketing — don't skip them.

Recommended flow:
  1. Freeze tasks + agent lineup (before running anything).
  2. Run harness/run_agent.py for each (agent x task), a few reps each.
  3. `tokenomist analyze runs/ --json reports.json` and
     `tokenomist trace runs/ --csv trace.csv`.
  4. Fill in the tables below and write the analysis.
-->

## The question

State the one decision this study informs, e.g. *"For <task type>, which model
gives the most correct solutions per dollar, and does a test-feedback scaffold
pay for itself?"* One or two sentences. No hedging.

## Setup

**Tasks (frozen before running).** Three self-contained coding tasks, each with
a hidden pytest suite as ground truth:

| Task id | What it asks | # tests | Why it's a fair test |
| --- | --- | --- | --- |
| `<task-1>` | … | … | edge cases: … |
| `<task-2>` | … | … | … |
| `<task-3>` | … | … | … |

**Agents.** <N> configurations from <your provider(s)>, spanning cost tiers plus
a raw-vs-scaffolded contrast:

| Agent | Price-book model | Scaffold (`--max-turns`) |
| --- | --- | --- |
| … | … | 6 (test-feedback) |
| … | … | 6 |
| … | … | 6 |
| `… (raw)` | … | 1 (single-shot) |

**Protocol.** Same `task.md`, system prompt, and turn cap for every agent.
`success_turn` = first assistant turn after which all tests pass. On failure the
harness feeds the pytest output back as a correction and the agent retries, up to
the cap. Token counts and latency are captured from the API (not estimated).
Each (agent × task) was run **<R> times**; numbers below are <mean / median>
with the spread noted. Prices from `prices.json`, `last_verified <date>`.

**What "correct" means.** All tests in a task's suite pass. This is the study's
ground truth and the thing Tokenomist itself does *not* verify — it's supplied
here by the harness.

## Results

### Leaderboard (pooled across tasks)

<!-- Paste/normalize `tokenomist analyze runs/`. -->

| Agent | Model | Solved | →Success | Tokens | Retries | Fixes | Latency | Cost | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| … | … | 3/3 | … | … | … | … | … | $… | … |

### The reframe: cost per *correct* solution

<!-- This is the headline. cost_per_correct = total_cost / #correct. An agent
that's cheap per run but often wrong can lose to a pricier, reliable one. -->

| Agent | Runs | # correct | Total cost | **Cost / correct solution** |
| --- | --- | --- | --- | --- |
| … | … | … | $… | **$…** |

### Per-task detail (optional)

Note any task where the ranking flipped, and why.

## **Where the metrics misled**

The most valuable section. Concrete examples where a headline number told the
wrong story, e.g.:

- An agent that "converged" fast (`success_turn` low) but on <task-x> produced a
  fix the extra edge-case test caught — high efficiency, wrong answer. Only the
  ground-truth suite exposed it.
- Convergence efficiency rewarded <agent> for few corrections, but it got there
  by <…>.
- Modeled vs measured: where captured latency/token cost diverged from what the
  price-book estimate would have said.

## **Limitations**

Be your own harshest reviewer:

- **n = <R> per cell.** Nondeterminism means these are indicative, not
  definitive. Observed spread: <…>.
- **Single domain** (small Python tasks). Results may not transfer to <other
  domains>.
- **Ground truth = my tests.** A passing suite isn't proof of a good solution,
  only of the properties I tested for.
- **Task selection.** I authored the tasks; a different set could shift rankings.
- **Prices** are public list rates as of `<date>` and exclude caching/batch
  discounts.

## Takeaway

One paragraph. What you'd actually do for a real project given this evidence —
and, honestly, where the data is too thin to say.

---

*Reproduce:* tasks in [`harness/tasks/`](../harness/tasks), runner in
[`harness/run_agent.py`](../harness/run_agent.py). Raw logs and
`reports.json` / `trace.csv` in `<where you put them>`.
