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
        logger.warning("generate_orders called with non-positive portfolio_value=%.2f", portfolio_value)
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
            logger.debug("generate_orders: skipping %s — Δweight %.4f < threshold %.4f",
                         ticker, abs(delta_value) / portfolio_value, min_trade_threshold)
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

        orders.append(Order(
            ticker=ticker,
            side=side,
            shares=delta_shares,
            estimated_price=price,
            estimated_value=estimated_value,
            reason=reason,
        ))
        logger.debug(
            "generate_orders: %s %s %d shares @ %.2f (Δ$%.0f)",
            side.upper(), ticker, delta_shares, price, delta_value,
        )

    return orders


def validate_orders_affordable(orders: list[Order], available_cash: float) -> bool:
    """Check that buy orders are covered by available cash plus sell proceeds.

    Assumes sells execute before buys (same-day closing price), so sell
    proceeds are immediately available to fund buys.

    Args:
        orders: Output of generate_orders.
        available_cash: Current CASH shares (== dollars) before any trades.

    Returns:
        True if affordable, False (and logs WARNING) if not.
    """
    sell_proceeds = sum(o.estimated_value for o in orders if o.side == "sell")
    buy_total = sum(o.estimated_value for o in orders if o.side == "buy")
    total_available = available_cash + sell_proceeds

    if total_available < buy_total:
        logger.warning(
            "validate_orders_affordable: INFEASIBLE — need $%.2f, have $%.2f "
            "(cash=%.2f + sells=%.2f)",
            buy_total, total_available, available_cash, sell_proceeds,
        )
        return False

    return True
