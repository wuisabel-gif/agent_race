# Changelog

All notable changes to Tokenomist are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Capture harness (`harness/run_agent.py`): provider-agnostic runner for
  OpenAI-compatible endpoints (Z.ai/GLM, DeepSeek, OpenAI, …) that solves a
  coding task with a test-feedback loop and emits native-format logs with real
  token usage and wall-clock latency. Includes an example task, an offline
  `--mock` mode, a `capture` extra, and a mock-mode smoke test.
- Case-study write-up scaffold (`docs/case-study.md`).
- Data-driven price book (`prices.json`) with input/output rates, cached-input
  rates, provider, and throughput per model family; verified 2026-07-04.
- Current-generation model prices across Anthropic, OpenAI, Google, Z.ai (GLM),
  DeepSeek, Mistral, MiniMax, Qwen, and Llama, alongside retained legacy models.
- Longest-family-prefix model matching, so dated ids like
  `claude-sonnet-4-6-20250514` resolve to their family.
- `--prices PATH` CLI flag and `PriceBook.from_file()` to override the bundled
  price book.
- `ModelPrice.cache_read_per_mtok`, `provider`, and `family` metadata; new
  `PriceBook.is_known()`.
- GitHub Actions CI: ruff lint + format check, and pytest across Python
  3.10–3.13.
- End-to-end CLI tests and expanded price-book tests.

### Changed
- Unknown models now report cost as `n/a` (via `None`) instead of a fabricated
  generic price; such reports sort after priced ones.
- Prices moved out of `pricing.py` into `prices.json`; the module now loads and
  matches against that data.
- Repository formatted with ruff.

## [0.1.0] - 2026-07-04

### Added
- Initial release: normalized conversation model; parsers for native,
  OpenAI/ChatGPT, Gemini, and LangGraph logs with format auto-detection;
  per-turn traffic-trace construction; aggregate agent metrics (cost, latency,
  retries, corrections, convergence efficiency); table/JSON/CSV reporting and
  ranking; CLI; optional Streamlit dashboard; and a zero-dependency browser demo.

[Unreleased]: https://github.com/wuisabel-gif/tokenomist/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wuisabel-gif/tokenomist/releases/tag/v0.1.0
