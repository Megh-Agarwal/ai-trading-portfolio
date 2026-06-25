"""Order generation for the M4 execution layer (Ticket 4.2).

Translates target weights from the optimizer into concrete share-level orders.
Purely arithmetic — risk logic already happened in M3.

Public API:
- Order: dataclass representing a single pending order.
- generate_orders: diff target vs current, produce a list of Orders.
- validate_orders_affordable: confirm sells cover buys given available cash.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

_CASH = "CASH"


@dataclass
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    shares: int
    estimated_price: float
    estimated_value: float  # shares × estimated_price (always positive)
    reason: str


def generate_orders(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    portfolio_value: float,
    prices: dict[str, float],
    min_trade_threshold: float = 0.001,
) -> list[Order]:
    """Translate target weights into share-level buy/sell orders.

    Iterates over the union of target_weights and current_positions tickers
    (excluding CASH). Skips tickers where the weight delta is below
    min_trade_threshold, or where the computed share count rounds down to zero.

    Shares are always rounded DOWN (floor) so we never spend more than the
    intended dollar amount — leftover cents accumulate in CASH.

    Args:
        target_weights: {ticker: weight} from the optimizer; CASH excluded.
        current_positions: {ticker: shares} from state.get_current_positions.
        portfolio_value: Total portfolio value in USD.
        prices: {ticker: closing_price} for the rebalance date.
        min_trade_threshold: Minimum |Δweight| to generate an order.

    Returns:
        List of Order objects, one per ticker requiring a trade.
    """
    if portfolio_value <= 0:
        logger.warning(
            "generate_orders called with non-positive portfolio_value=%.2f", portfolio_value
        )
        return []

    tickers = (set(target_weights.keys()) | set(current_positions.keys())) - {_CASH}
    orders: list[Order] = []

    for ticker in sorted(tickers):
        price = prices.get(ticker)
        if price is None or price <= 0:
            logger.warning("generate_orders: skipping %s — missing or non-positive price", ticker)
            continue

        target_weight = target_weights.get(ticker, 0.0)
        target_value = target_weight * portfolio_value

        current_shares = current_positions.get(ticker, 0.0)
        current_value = current_shares * price

        delta_value = target_value - current_value

        if abs(delta_value) / portfolio_value < min_trade_threshold:
            logger.debug(
                "generate_orders: skipping %s — Δweight %.4f < threshold %.4f",
                ticker,
                abs(delta_value) / portfolio_value,
                min_trade_threshold,
            )
            continue

        delta_shares = math.floor(abs(delta_value) / price)
        if delta_shares == 0:
            logger.debug("generate_orders: skipping %s — delta rounds to 0 shares", ticker)
            continue

        side: Literal["buy", "sell"] = "buy" if delta_value > 0 else "sell"
        estimated_value = delta_shares * price
        current_pct = (current_value / portfolio_value) * 100
        target_pct = target_weight * 100
        reason = f"target {target_pct:.1f}% vs current {current_pct:.1f}%"

        orders.append(
            Order(
                ticker=ticker,
                side=side,
                shares=delta_shares,
                estimated_price=price,
                estimated_value=estimated_value,
                reason=reason,
            )
        )
        logger.debug(
            "generate_orders: %s %s %d shares @ %.2f (Δ$%.0f)",
            side.upper(),
            ticker,
            delta_shares,
            price,
            delta_value,
        )

    return orders


def validate_orders_affordable(
    orders: list[Order],
    available_cash: float,
    cost_rate: float = 0.0,
) -> list[Order]:
    """Ensure buy orders can be funded by available cash and net sell proceeds.

    Uses cost-aware arithmetic:
      funds available  = cash + gross_sell_value × (1 − cost_rate)
      required capital = gross_buy_value  × (1 + cost_rate)

    If the rebalance is unaffordable, all buy orders are proportionally scaled
    down (flooring share counts to keep them integral) until the constraint is
    satisfied.  A WARNING is logged with the scale factor.  Sell orders are
    never modified.

    Args:
        orders: Output of generate_orders (sells should precede buys).
        available_cash: Current CASH shares (== dollars) before any trades.
        cost_rate: One-way transaction cost as a decimal (e.g. 0.001 for 10 bps).
                   Defaults to 0.0 for backward compatibility with call sites
                   that do not have a config available.

    Returns:
        List of orders, with buy quantities possibly reduced.  Sell orders and
        orders that remain affordable are returned unchanged.
    """
    gross_sells = sum(o.estimated_value for o in orders if o.side == "sell")
    gross_buys = sum(o.estimated_value for o in orders if o.side == "buy")

    funds_available = available_cash + gross_sells * (1.0 - cost_rate)
    total_buy_cost = gross_buys * (1.0 + cost_rate)

    if gross_buys == 0 or total_buy_cost <= funds_available:
        return orders

    if funds_available <= 0:
        logger.warning(
            "validate_orders_affordable: no funds available after sell costs "
            "(cash=%.2f, net_sells=%.2f) — dropping all buy orders",
            available_cash,
            gross_sells * (1.0 - cost_rate),
        )
        return [o for o in orders if o.side == "sell"]

    scale = funds_available / total_buy_cost
    logger.warning(
        "validate_orders_affordable: scaling buy orders by %.6f "
        "(need $%.2f incl. costs, have $%.2f after sell costs)",
        scale,
        total_buy_cost,
        funds_available,
    )

    scaled: list[Order] = []
    for o in orders:
        if o.side != "buy":
            scaled.append(o)
            continue
        new_shares = math.floor(o.shares * scale)
        if new_shares == 0:
            logger.warning(
                "validate_orders_affordable: %s scaled to 0 shares — order dropped",
                o.ticker,
            )
            continue
        scaled.append(
            Order(
                ticker=o.ticker,
                side=o.side,
                shares=new_shares,
                estimated_price=o.estimated_price,
                estimated_value=new_shares * o.estimated_price,
                reason=o.reason + f" [scaled ×{scale:.4f}]",
            )
        )

    return scaled
