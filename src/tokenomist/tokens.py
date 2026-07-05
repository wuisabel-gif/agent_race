"""Token estimation.

Real vendor logs rarely include token counts on every turn, so we need a
fallback. If ``tiktoken`` is installed we use it for a faithful BPE count;
otherwise we fall back to a well-known heuristic (~4 characters per token for
English text, with a small adjustment for whitespace/structure).

The heuristic is intentionally simple and deterministic so that test results
and the dashboard are reproducible on any machine, with or without optional
dependencies.
"""

from __future__ import annotations

import functools

_CHARS_PER_TOKEN = 4.0


@functools.lru_cache(maxsize=1)
def _tiktoken_encoder():  # pragma: no cover - depends on optional dependency
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str | None) -> int:
    """Estimate the number of tokens in ``text``.

    Uses ``tiktoken`` when available, otherwise a character-based heuristic.
    """

    if not text:
        return 0

    encoder = _tiktoken_encoder()
    if encoder is not None:  # pragma: no cover - optional path
        return len(encoder.encode(text))

    # Heuristic: ~4 chars/token, but never report 0 tokens for non-empty text.
    return max(1, round(len(text) / _CHARS_PER_TOKEN))
