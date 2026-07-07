"""Tests for token estimation and the pricing/latency models."""

from __future__ import annotations

import json

from tokenomist.pricing import DEFAULT_PRICES, ModelPrice, PriceBook
from tokenomist.tokens import estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_nonzero():
    assert estimate_tokens("hello world") >= 1
    # Longer text yields more tokens.
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_pricebook_resolve_exact_and_family():
    book = PriceBook()
    # Exact family id.
    assert book.resolve("claude-opus-4-8") is DEFAULT_PRICES["claude-opus-4-8"]
    # Dated suffix resolves to the family via longest-prefix match.
    dated = book.resolve("claude-sonnet-4-6-20250514")
    assert dated is DEFAULT_PRICES["claude-sonnet-4-6"]
    # Short alias resolves to a current model.
    assert book.resolve("claude-opus") is DEFAULT_PRICES["claude-opus-4-8"]


def test_longest_family_prefix_wins():
    book = PriceBook()
    # "claude-opus-4-8-x" should pick the 4-8 family, not a shorter opus alias.
    price = book.resolve("claude-opus-4-8-experimental")
    assert price is DEFAULT_PRICES["claude-opus-4-8"]


def test_model_patterns_resolve_provider_model_strings():
    book = PriceBook()
    assert book.resolve("zhipu/glm-5.1") is DEFAULT_PRICES["glm-5.1"]
    assert book.resolve("deepseek/deepseek-chat") is DEFAULT_PRICES["deepseek-v4-pro"]
    assert book.resolve("deepseek-v3.2") is DEFAULT_PRICES["deepseek-v3.2"]


def test_pricebook_unknown_returns_none():
    book = PriceBook()
    assert book.resolve("totally-unknown-model") is None
    assert book.cost_usd("totally-unknown-model", 1000, 1000) is None
    assert book.is_known("totally-unknown-model") is False
    assert book.is_known("claude-opus-4-8") is True


def test_unknown_model_still_estimates_latency():
    # Latency is not a billing claim, so it falls back to a default throughput.
    book = PriceBook()
    assert book.latency_ms("totally-unknown-model", 1000) > 0


def test_cache_read_rate_present_for_major_models():
    book = PriceBook()
    price = book.resolve("claude-sonnet-4-6")
    assert price is not None
    assert price.cache_read_per_mtok == 0.3


def test_cost_scales_with_output():
    book = PriceBook()
    cheap = book.cost_usd("gpt-4o", 1000, 0)
    pricier = book.cost_usd("gpt-4o", 1000, 1000)
    assert cheap is not None and pricier is not None
    assert pricier > cheap


def test_cost_details_include_cache_read():
    book = PriceBook()
    details = book.cost_details_usd(
        "gpt-4o",
        {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 100},
    )
    assert details is not None
    assert details["input"] == 800 / 1_000_000 * 2.5
    assert details["cache_read"] == 200 / 1_000_000 * 1.25
    assert details["output"] == 100 / 1_000_000 * 10.0
    assert details["total"] == sum(value for key, value in details.items() if key != "total")


def test_latency_grows_with_output():
    book = PriceBook()
    assert book.latency_ms("gpt-4o", 1000) > book.latency_ms("gpt-4o", 10)


def test_custom_pricebook_overrides():
    book = PriceBook({"mymodel": ModelPrice(1.0, 2.0, 50.0)})
    assert book.cost_usd("mymodel", 1_000_000, 0) == 1.0
    # A dict-constructed book only knows what it was given.
    assert book.resolve("claude-opus-4-8") is None


def test_pricebook_from_file(tmp_path):
    data = {
        "default_output_tokens_per_sec": 42,
        "models": [
            {
                "family": "acme-1",
                "input": 2.0,
                "output": 4.0,
                "cache_read": 0.2,
                "tps": 99,
                "aliases": ["acme"],
                "model_patterns": ["^provider/acme$"],
            },
        ],
    }
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    book = PriceBook.from_file(path)
    assert book.cost_usd("acme-1", 1_000_000, 0) == 2.0
    # Alias and dated suffix both resolve.
    assert book.resolve("acme") is not None
    assert book.resolve("acme-1-20260101") is not None
    assert book.resolve("provider/acme") is not None
    # Unknown-to-this-file model is n/a.
    assert book.cost_usd("claude-opus-4-8", 1000, 1000) is None
