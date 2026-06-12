"""Token pricing for Anthropic models used in this project.

Prices are USD per 1 million tokens. Verify against https://www.anthropic.com/pricing
before reporting cost figures in the paper.

Last verified: 2026-06-12
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (input_price_per_mtok, output_price_per_mtok) in USD
_PRICES: dict[str, tuple[float, float]] = {
    # Claude Haiku 4.5 — used for news sentiment agent (high-volume, low-cost)
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),
    # Claude Sonnet 4.6 — used for macro regime and events agents
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    # Claude Opus 4.7 — not used in this project but listed for completeness
    "claude-opus-4-7": (15.00, 75.00),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate USD cost for a single Anthropic API call.

    Args:
        model: Model string exactly as passed to the API (e.g. 'claude-haiku-4-5-20251001').
        tokens_in: Input token count from response.usage.input_tokens.
        tokens_out: Output token count from response.usage.output_tokens.

    Returns:
        Cost in USD. Returns 0.0 for unrecognised models (with a warning).
    """
    rates = _PRICES.get(model)
    if rates is None:
        # Try prefix match (e.g. 'claude-haiku-4-5-...' → 'claude-haiku-4-5')
        for key, prices in _PRICES.items():
            if model.startswith(key):
                rates = prices
                break

    if rates is None:
        logger.warning("No pricing data for model %r — cost recorded as $0.00", model)
        return 0.0

    input_rate, output_rate = rates
    return (tokens_in * input_rate + tokens_out * output_rate) / 1_000_000


def model_rates(model: str) -> tuple[float, float] | None:
    """Return (input_rate, output_rate) per million tokens, or None if unknown."""
    rates = _PRICES.get(model)
    if rates:
        return rates
    for key, prices in _PRICES.items():
        if model.startswith(key):
            return prices
    return None
