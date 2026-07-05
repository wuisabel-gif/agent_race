# Log format specification

Tokenomist analyzes **conversation logs** — one multi-turn run of one agent
solving one task. This document is the exact contract for what you upload (in
the browser demo) or pass to the CLI/library.

## Rules of thumb

- **One JSON file = one agent, one task, one run.**
- **To compare agents,** give every file the **same `task_id`** and load them
  together (the demo accepts multi-select and drag-and-drop; the CLI accepts a
  glob or directory).
- **The format is auto-detected** from the JSON shape — you never select it.
  You *can* force it in the CLI with `--format`.
- The file must be a single JSON **object** (not an array of conversations).

---

## Top-level fields

These are the metadata you supply. The three "answer key" fields
(`success_turn`, `final_correct`, `final_score`) are what let the tool score a
run — without them a run shows as *failed*, efficiency `0`, and its car never
finishes the race.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `agent` | string | recommended | Display name (e.g. `"Claude"`, `"ChatGPT"`). Defaults per format. |
| `model` | string | recommended | Drives cost & latency estimation. Use a known id (below). |
| `provider` | string | optional | Free-form label (`"anthropic"`, `"openai"`, …). |
| `task_id` | string | recommended | Identifies the task. **Same value across files = same task compared.** |
| `success_turn` | int \| null | **yes\*** | **0-based index** into the turns array of the first turn that produced a useful/correct answer. `null` = never converged. |
| `final_correct` | bool | **yes\*** | Was the final result actually correct. |
| `final_score` | float `0.0–1.0` | **yes\*** | Quality of the final answer. `1.0` = perfect, `0.0` = failed. Sets the result badge (`correct` / `partial` / `failed`). |

\* Technically optional (they default to `null` / `false` / `0.0`), but a run
with them omitted is treated as an unsolved/failed run.

### How the fields become metrics

- `success_turn` → **turns-to-success**, **tokens-to-success**, and is the
  cutoff for convergence efficiency.
- `final_score` + corrections + tokens-to-success → **convergence efficiency**
  (`final_score × budget_term × 1/(1+corrections)`), the value the race and the
  default ranking use.
- `is_correction` user turns → **correction count** (auto-detected from phrasing
  if you don't set the flag).
- `is_retry` assistant turns → **retry loops**.
- `tool_calls[].ok` → **tool success rate**.

### Known model ids (for accurate pricing)

The price book lives in [`src/tokenomist/prices.json`](src/tokenomist/prices.json)
and covers current models across Anthropic, OpenAI, Google, Z.ai (GLM),
DeepSeek, Mistral, MiniMax, Qwen, and Llama, plus retained legacy ids.

Matching is by **longest family prefix**, so a dated id like
`claude-sonnet-4-6-20250514` resolves to the `claude-sonnet-4-6` family, and
short aliases (`claude-sonnet`, `glm-5.1`, `gpt-5`) resolve to a current model.
An **unknown model reports cost as `n/a`** rather than a fabricated rate. Prices
are approximate public list rates for *relative* comparison, not billing;
override them with your own file via `--prices` (CLI) or `PriceBook.from_file()`
(library). See the price book for the full, current list of ids.

---

## Per-turn fields

| Field | Type | Meaning |
| --- | --- | --- |
| `role` (or `type` for LangGraph) | string | `user` / `assistant` / `tool` / `system`. LangGraph uses `human` / `ai` / `tool` / `system`. |
| `content` | string | The message text. |
| `input_tokens`, `output_tokens` | int | Optional. If omitted, estimated from the text (~4 chars/token, or tiktoken if installed in the Python lib). |
| `latency_ms` | number | Optional. If omitted, estimated from output tokens. |
| `tool_calls` | array | Optional. See below. |
| `is_correction` | bool | Optional. Marks a **user** turn that corrected the agent. Auto-detected from phrasing (`"no, that's wrong"`, `"try again"`, …) if omitted. |
| `is_retry` | bool | Optional. Marks an **assistant** turn that retried after a failure. Auto-detected if omitted. |

### Tool call object

```json
{ "name": "run_tests", "arguments": { "path": "test_x.py" }, "ok": true, "result": "5 passed" }
```

- `name` — tool name (string).
- `arguments` — object (OpenAI may send a JSON-encoded **string**; that's parsed
  automatically).
- `ok` — did the call succeed (default `true`). `false` lowers tool success rate.
- `result` — optional string output.

---

## The four accepted shapes

Detection key: **native** has `turns`; **LangGraph** has `messages[].type`;
**OpenAI** has `messages[].role`; **Gemini** has `contents[].parts`.

### 1. Native (recommended — richest)

```json
{
  "agent": "Claude",
  "model": "claude-sonnet",
  "task_id": "fix-login-bug",
  "success_turn": 2,
  "final_correct": true,
  "final_score": 1.0,
  "turns": [
    { "role": "user", "content": "The login form throws on empty email. Fix it." },
    { "role": "assistant", "content": "Reading the handler.",
      "tool_calls": [ { "name": "read_file", "arguments": { "path": "login.js" }, "ok": true } ] },
    { "role": "assistant", "content": "Added a guard for empty email; tests pass.",
      "output_tokens": 40 }
  ]
}
```

### 2. OpenAI / ChatGPT / Agents SDK

`messages` with `role`; assistant `tool_calls` use a `function` wrapper whose
`arguments` may be a JSON string; tool output is a `role: "tool"` message.

```json
{
  "agent": "ChatGPT",
  "model": "gpt-4o",
  "task_id": "fix-login-bug",
  "success_turn": 3,
  "final_correct": true,
  "final_score": 1.0,
  "messages": [
    { "role": "system", "content": "You are a coding assistant." },
    { "role": "user", "content": "Fix the empty-email crash." },
    { "role": "assistant", "content": "Patching.",
      "tool_calls": [ { "function": { "name": "python", "arguments": "{\"code\": \"...\"}" }, "ok": true } ] },
    { "role": "tool", "content": "tests pass" },
    { "role": "assistant", "content": "Fixed and verified." }
  ]
}
```

### 3. Gemini (`generateContent`)

`contents` with `role` (`user` / `model`) and a `parts` list; function calls are
`{"functionCall": {...}}` parts, results are `{"functionResponse": {...}}`.

```json
{
  "agent": "Gemini",
  "model": "gemini-1.5-pro",
  "task_id": "fix-login-bug",
  "success_turn": 1,
  "final_correct": true,
  "final_score": 1.0,
  "contents": [
    { "role": "user", "parts": [ { "text": "Fix the empty-email crash." } ] },
    { "role": "model", "parts": [
      { "text": "Adding a guard." },
      { "functionCall": { "name": "run_code", "args": { "code": "..." } } },
      { "functionResponse": { "name": "run_code", "response": "ok" } }
    ] }
  ]
}
```

### 4. LangGraph / LangChain

`messages` with a `type` discriminator; AI `tool_calls` use `args`; tool output
is a `type: "tool"` message.

```json
{
  "agent": "LangGraph Agent",
  "model": "gpt-4o-mini",
  "task_id": "fix-login-bug",
  "success_turn": 3,
  "final_correct": true,
  "final_score": 1.0,
  "messages": [
    { "type": "human", "content": "Fix the empty-email crash." },
    { "type": "ai", "content": "Inspecting.", "tool_calls": [ { "name": "read_file", "args": { "path": "login.js" } } ] },
    { "type": "tool", "content": "function handleLogin(email) { ... }" },
    { "type": "ai", "content": "Added a guard; CI is green." }
  ]
}
```

---

## Common mistakes

- **Forgetting `success_turn` / `final_score`** → the run looks failed. Set them.
- **`success_turn` off by one** → it indexes the **whole** turns/messages/contents
  array (including `user`, `system`, and `tool` turns), 0-based.
- **Different `task_id` per file** → agents won't be presented as the same race.
- **Wrapping multiple conversations in one array** → upload separate files instead.
- **Unknown `model` string** → pricing falls back to a generic rate; use a known id.
