"""Cost and latency models.

Prices live in a data file (:data:`prices.json` next to this module) rather than
in code, so they can be updated without touching logic and overridden per run.
Each entry is a *family* — a stable model-name prefix such as ``claude-sonnet-4-6``
— plus USD-per-million-token input/output rates, an optional cached-input rate,
and a rough output throughput used to estimate latency when a log has no
timestamps. The numbers are approximate public list prices meant for *relative*
comparison between agents, not for billing.

Model names in real logs carry dated suffixes (``claude-sonnet-4-6-20250514``),
so lookup matches by longest family prefix, falling back to short aliases and
finally to ``None`` — an unknown model yields ``cost = n/a`` rather than a
silently fabricated number.
"""

from __future__ import annotations

import json
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


def _load_price_records(path: str | Path | None = None) -> tuple[dict[str, ModelPrice], float]:
    """Return ``(family/alias -> ModelPrice, default_tps)`` from a JSON file.

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
    for rec in data.get("models", []):
        family = str(rec["family"]).lower()
        price = ModelPrice(
            input_per_mtok=float(rec["input"]),
            output_per_mtok=float(rec["output"]),
            output_tokens_per_sec=float(rec.get("tps", default_tps)),
            cache_read_per_mtok=(
                None if rec.get("cache_read") is None else float(rec["cache_read"])
            ),
            provider=rec.get("provider"),
            family=family,
        )
        table[family] = price
        for alias in rec.get("aliases", []):
            table.setdefault(str(alias).lower(), price)
    return table, default_tps


# Bundled default price book, exposed for inspection and tests.
DEFAULT_PRICES, _DEFAULT_TPS = _load_price_records()


class PriceBook:
    """Lookup of model name -> :class:`ModelPrice` with prefix/family matching."""

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        # An explicit dict (e.g. from tests or a caller) is used as-is; otherwise
        # the bundled default price book is loaded.
        self._prices = dict(DEFAULT_PRICES if prices is None else prices)

    @classmethod
    def from_file(cls, path: str | Path) -> PriceBook:
        """Build a price book from a user-supplied JSON file (see prices.json)."""

        table, _ = _load_price_records(path)
        return cls(table)

    def resolve(self, model: str | None) -> ModelPrice | None:
        """Return the price for ``model``, or ``None`` if it is unknown.

        Matching order: exact (case-insensitive) id/alias, then the longest
        known family that is a prefix of the model name (so dated suffixes like
        ``-20250514`` still match), then ``None``.
        """

        if not model:
            return None
        key = model.lower().strip()
        exact = self._prices.get(key)
        if exact is not None:
            return exact

        # Longest family that is a prefix of the queried model name wins.
        best: ModelPrice | None = None
        best_len = -1
        for name, price in self._prices.items():
            if key.startswith(name) and len(name) > best_len:
                best, best_len = price, len(name)
        return best

    def is_known(self, model: str | None) -> bool:
        return self.resolve(model) is not None

    def cost_usd(self, model: str | None, input_tokens: int, output_tokens: int) -> float | None:
        """Token cost in USD, or ``None`` when the model's price is unknown."""

        price = self.resolve(model)
        if price is None:
            return None
        return (
            input_tokens / 1_000_000 * price.input_per_mtok
            + output_tokens / 1_000_000 * price.output_per_mtok
        )

    def latency_ms(self, model: str | None, output_tokens: int) -> float:
        """Estimate generation latency from output token count.

        Latency is not a billing claim, so an unknown model falls back to a
        default throughput rather than returning ``None``.
        """

        price = self.resolve(model)
        tps = (price.output_tokens_per_sec if price else _DEFAULT_TPS) or _DEFAULT_TPS
        return _TURN_OVERHEAD_MS + (output_tokens / tps) * 1000.0
