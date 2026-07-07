"""Cost and latency models.

Prices live in a data file (:data:`prices.json` next to this module) rather than
in code, so they can be updated without touching logic and overridden per run.
Each entry is a *family* — a stable model-name prefix such as ``claude-sonnet-4-6``
— plus aliases, optional regex model patterns, USD-per-million-token
input/output rates, an optional cached-input rate, and a rough output throughput
used to estimate latency when a log has no timestamps. The numbers are
approximate public list prices meant for *relative* comparison between agents,
not for billing.

Model names in real logs carry dated suffixes, provider prefixes, and endpoint
aliases (``claude-sonnet-4-6-20250514``, ``zhipu/glm-5.1``,
``deepseek-chat``), so lookup matches exact ids/aliases, longest family prefix,
then regex model patterns. An unknown model yields ``cost = n/a`` rather than a
silently fabricated number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.9+: read bundled data via importlib.resources
    from importlib.resources import files as _pkg_files
except ImportError:  # pragma: no cover
    _pkg_files = None

_PRICES_FILENAME = "prices.json"

_DEFAULT_OUTPUT_TPS = 60.0
# Fixed per-turn overhead added to the throughput-derived latency estimate.
_TURN_OVERHEAD_MS = 350.0


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices, optional cache rate, and rough throughput."""

    input_per_mtok: float
    output_per_mtok: float
    output_tokens_per_sec: float = _DEFAULT_OUTPUT_TPS
    cache_read_per_mtok: float | None = None
    provider: str | None = None
    family: str | None = None
    model_patterns: tuple[str, ...] = ()


def _load_price_records(
    path: str | Path | None = None,
) -> tuple[dict[str, ModelPrice], list[tuple[re.Pattern[str], ModelPrice]], float]:
    """Return ``(family/alias table, regex matchers, default_tps)`` from JSON.

    When ``path`` is ``None`` the price book bundled with the package is used.
    """

    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
    elif _pkg_files is not None:
        raw = (_pkg_files("tokenomist") / _PRICES_FILENAME).read_text(encoding="utf-8")
    else:  # pragma: no cover - fallback for exotic environments
        raw = (Path(__file__).parent / _PRICES_FILENAME).read_text(encoding="utf-8")

    data: dict[str, Any] = json.loads(raw)
    default_tps = float(data.get("default_output_tokens_per_sec", _DEFAULT_OUTPUT_TPS))

    table: dict[str, ModelPrice] = {}
    matchers: list[tuple[re.Pattern[str], ModelPrice]] = []
    for rec in data.get("models", []):
        family = str(rec["family"]).lower()
        patterns = tuple(
            str(pattern) for pattern in (rec.get("model_patterns") or [])
        )
        price = ModelPrice(
            input_per_mtok=float(rec["input"]),
            output_per_mtok=float(rec["output"]),
            output_tokens_per_sec=float(rec.get("tps", default_tps)),
            cache_read_per_mtok=(
                None if rec.get("cache_read") is None else float(rec["cache_read"])
            ),
            provider=rec.get("provider"),
            family=family,
            model_patterns=patterns,
        )
        table[family] = price
        for alias in rec.get("aliases", []):
            table.setdefault(str(alias).lower(), price)
        for pattern in patterns:
            matchers.append((re.compile(pattern, flags=re.IGNORECASE), price))
    return table, matchers, default_tps


# Bundled default price book, exposed for inspection and tests.
DEFAULT_PRICES, _DEFAULT_MATCHERS, _DEFAULT_TPS = _load_price_records()


class PriceBook:
    """Lookup of model name -> :class:`ModelPrice` with prefix/family matching."""

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        # An explicit dict (e.g. from tests or a caller) is used as-is; otherwise
        # the bundled default price book is loaded.
        self._prices = dict(DEFAULT_PRICES if prices is None else prices)
        self._matchers = list(_DEFAULT_MATCHERS if prices is None else [])

    @classmethod
    def from_file(cls, path: str | Path) -> PriceBook:
        """Build a price book from a user-supplied JSON file (see prices.json)."""

        table, matchers, _ = _load_price_records(path)
        book = cls(table)
        book._matchers = matchers
        return book

    def resolve(self, model: str | None) -> ModelPrice | None:
        """Return the price for ``model``, or ``None`` if it is unknown.

        Matching order: exact (case-insensitive) id/alias, then the longest
        known family that is a prefix of the model name (so dated suffixes like
        ``-20250514`` still match), then regex match patterns from the price
        catalog, then ``None``.
        """

        if not model:
            return None
        key = model.lower().strip()
        exact = self._prices.get(key)
        if exact is not None:
            return exact

        # Longest canonical family that is a prefix of the queried model name
        # wins. Short aliases are exact-match only; otherwise an alias like
        # "deepseek" would swallow provider strings before regex patterns run.
        best: ModelPrice | None = None
        best_len = -1
        for name, price in self._prices.items():
            if name == price.family and key.startswith(name) and len(name) > best_len:
                best, best_len = price, len(name)
        if best is not None:
            return best

        for pattern, price in self._matchers:
            if pattern.search(model.strip()):
                return price
        return best

    def is_known(self, model: str | None) -> bool:
        return self.resolve(model) is not None

    def cost_usd(self, model: str | None, input_tokens: int, output_tokens: int) -> float | None:
        """Token cost in USD, or ``None`` when the model's price is unknown."""

        details = self.cost_details_usd(
            model,
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return None if details is None else details.get("total", sum(details.values()))

    def cost_details_usd(
        self, model: str | None, usage_details: dict[str, int]
    ) -> dict[str, float] | None:
        """Cost map in USD by usage dimension, or ``None`` for unknown models."""

        price = self.resolve(model)
        if price is None:
            return None

        details: dict[str, float] = {}
        input_tokens = usage_details.get("input_tokens", usage_details.get("input", 0))
        output_tokens = usage_details.get("output_tokens", usage_details.get("output", 0))
        cached_tokens = usage_details.get("cached_input_tokens", 0) + usage_details.get(
            "cache_read_input_tokens", 0
        )

        billable_input_tokens = max(0, input_tokens - cached_tokens)
        if billable_input_tokens:
            details["input"] = billable_input_tokens / 1_000_000 * price.input_per_mtok
        if output_tokens:
            details["output"] = output_tokens / 1_000_000 * price.output_per_mtok
        if cached_tokens and price.cache_read_per_mtok is not None:
            details["cache_read"] = cached_tokens / 1_000_000 * price.cache_read_per_mtok

        if details:
            details["total"] = sum(details.values())
        return details

    def latency_ms(self, model: str | None, output_tokens: int) -> float:
        """Estimate generation latency from output token count.

        Latency is not a billing claim, so an unknown model falls back to a
        default throughput rather than returning ``None``.
        """

        price = self.resolve(model)
        tps = (price.output_tokens_per_sec if price else _DEFAULT_TPS) or _DEFAULT_TPS
        return _TURN_OVERHEAD_MS + (output_tokens / tps) * 1000.0
