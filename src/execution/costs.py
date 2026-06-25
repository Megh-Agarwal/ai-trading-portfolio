"""Transaction cost model for sector ETF trades (ADR-017).

One-way cost = trade_value × (spread_bps + slippage_bps) / 10_000

All bps values are read from config — no hardcoded constants in this module.

Public API:
- estimate_trade_cost: cost in USD for a single trade.
- estimate_portfolio_rebalance_cost: total cost for a full rebalance.
- compute_cost_drag_bps: express a USD cost as portfolio bps for attribution.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cost_rate(config) -> float:
    """One-way cost per dollar traded: (spread_bps + slippage_bps) / 10_000."""
    return (config.spread_bps + config.slippage_bps) / 10_000.0


def estimate_trade_cost(
    ticker: str,
    trade_value_usd: float,
    config,  # TransactionCostsConfig
) -> float:
    """Estimate the one-way transaction cost for a single trade.

    Cost = trade_value_usd × (spread_bps + slippage_bps) / 10_000

    The ticker parameter is accepted for API consistency and future per-ticker
    bps differentiation (v2). In v1 all sector ETFs share the same bps rates.

    Args:
        ticker: Sector ETF ticker symbol (e.g. "XLK"). Unused in v1.
        trade_value_usd: Absolute dollar value of the trade (always positive).
        config: TransactionCostsConfig from optimizer.yaml.

    Returns:
        Cost in USD (non-negative float).
    """
    return trade_value_usd * _cost_rate(config)


def estimate_portfolio_rebalance_cost(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    portfolio_value: float,
    config,  # TransactionCostsConfig
) -> float:
    """Total one-way transaction cost for a portfolio rebalance.

    Only sectors where |Δw| > config.min_trade_threshold are charged.
    Positions at or below the threshold are assumed to be rounding noise
    and incur no cost.

    Args:
        old_weights: Current weight vector, shape (n,).
        new_weights: Target weight vector, shape (n,).
        portfolio_value: Total portfolio value in USD.
        config: TransactionCostsConfig from optimizer.yaml.

    Returns:
        Total cost in USD across all traded sectors.
    """
    threshold = config.min_trade_threshold
    rate = _cost_rate(config)
    total_cost = 0.0

    for delta_w in np.abs(new_weights - old_weights):
        if delta_w > threshold:
            total_cost += float(delta_w) * portfolio_value * rate

    logger.debug(
        "estimate_portfolio_rebalance_cost  portfolio=$%.0f  cost=$%.2f  (%.2f bps drag)",
        portfolio_value,
        total_cost,
        compute_cost_drag_bps(total_cost, portfolio_value) if portfolio_value > 0 else 0,
    )
    return total_cost


def compute_cost_drag_bps(cost_usd: float, portfolio_value: float) -> float:
    """Express a USD cost as basis points of portfolio value.

    Used in performance attribution to report transaction cost drag on
    annualised returns.

    Args:
        cost_usd: Cost in USD (non-negative).
        portfolio_value: Total portfolio value in USD (must be > 0).

    Returns:
        Cost as basis points (1 bp = 0.01% of portfolio).
    """
    return cost_usd / portfolio_value * 10_000.0
