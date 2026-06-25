"""Fill simulation for the M4 execution layer (Ticket 4.3).

Applies orders from 4.2 using the cost model from 3.4, writes trade records,
and updates portfolio state via 4.1's write_positions.

Transaction cost convention:
  Buy:  net_value = gross_value + cost_usd  (costs you more)
  Sell: net_value = gross_value - cost_usd  (you receive less)

Slippage is already baked into cost_usd via estimate_trade_cost — the Trade
row stores 0.0 in the slippage column to avoid double-counting.
"""

from __future__ import annotations

import datetime
import logging

from sqlalchemy.orm import Session

from db.models import PORTFOLIO_LIVE, Trade
from exceptions import NegativeCashError
from execution.costs import estimate_trade_cost
from execution.orders import Order
from execution.state import write_positions

logger = logging.getLogger(__name__)

_CASH = "CASH"


def simulate_fill(order: Order, config) -> dict:
    """Compute the fill result for a single order.

    Cost is computed via estimate_trade_cost (Ticket 3.4). Fill price equals
    the order's estimated closing price — no separate slippage model.

    Args:
        order: Order from generate_orders.
        config: TransactionCostsConfig from optimizer.yaml.

    Returns:
        Dict with keys: ticker, side, shares, fill_price, gross_value,
        cost_usd, net_value.
    """
    gross_value = order.shares * order.estimated_price
    cost_usd = estimate_trade_cost(order.ticker, gross_value, config)
    net_value = gross_value + cost_usd if order.side == "buy" else gross_value - cost_usd

    return {
        "ticker": order.ticker,
        "side": order.side,
        "shares": order.shares,
        "fill_price": order.estimated_price,
        "gross_value": gross_value,
        "cost_usd": cost_usd,
        "net_value": net_value,
    }


def simulate_all_fills(
    orders: list[Order],
    date: str,
    db: Session,
    config,  # TransactionCostsConfig
    portfolio_id: str = PORTFOLIO_LIVE,
) -> list[dict]:
    """Simulate fills for all orders and write trade records to the DB.

    Processes orders in the order given (callers should arrange sells before
    buys so the affordability logic in 4.2 stays consistent).

    Args:
        orders: List of Order objects from generate_orders.
        date: ISO date string (YYYY-MM-DD) for the trade records.
        db: SQLAlchemy session.
        config: TransactionCostsConfig from optimizer.yaml.
        portfolio_id: Portfolio namespace for the trade rows.

    Returns:
        List of fill dicts (one per order) for downstream apply_fills_to_state.
    """
    date_obj = datetime.date.fromisoformat(date)
    fills: list[dict] = []

    for order in orders:
        fill = simulate_fill(order, config)
        db.add(
            Trade(
                portfolio_id=portfolio_id,
                date=date_obj,
                ticker=fill["ticker"],
                side=fill["side"],
                shares=fill["shares"],
                price=fill["fill_price"],
                commission=fill["cost_usd"],
                slippage=0.0,
            )
        )
        fills.append(fill)
        logger.debug(
            "simulate_fill: %s %s %d @ %.2f  gross=$%.2f  cost=$%.4f  net=$%.2f",
            fill["side"].upper(),
            fill["ticker"],
            fill["shares"],
            fill["fill_price"],
            fill["gross_value"],
            fill["cost_usd"],
            fill["net_value"],
        )

    db.commit()
    return fills


def apply_fills_to_state(
    date: str,
    fills: list[dict],
    current_positions: dict[str, float],
    db: Session,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict[str, float]:
    """Apply fill results to portfolio positions and persist the new state.

    Buys increase the ticker position and decrease CASH by net_value.
    Sells decrease the ticker position and increase CASH by net_value.
    Raises NegativeCashError if CASH goes below zero after all fills.

    Args:
        date: ISO date string (YYYY-MM-DD) for the updated position rows.
        fills: List of fill dicts from simulate_all_fills.
        current_positions: {ticker: shares} snapshot before any fills.
        db: SQLAlchemy session.
        portfolio_id: Portfolio namespace for the position writes.

    Returns:
        Updated positions dict after all fills are applied.

    Raises:
        NegativeCashError: If CASH is negative after fills, indicating the
            affordability check was bypassed or fill math has a bug.
    """
    positions = dict(current_positions)

    for fill in fills:
        ticker = fill["ticker"]
        shares = float(fill["shares"])
        net_value = fill["net_value"]

        if fill["side"] == "buy":
            positions[ticker] = positions.get(ticker, 0.0) + shares
            positions[_CASH] = positions.get(_CASH, 0.0) - net_value
        else:
            positions[ticker] = positions.get(ticker, 0.0) - shares
            positions[_CASH] = positions.get(_CASH, 0.0) + net_value

    cash_balance = positions.get(_CASH, 0.0)
    if cash_balance < 0:
        raise NegativeCashError(
            f"CASH went negative after fills: {cash_balance:.4f}. "
            "Affordability check must have been bypassed or fill math is incorrect."
        )

    write_positions(date, positions, db, portfolio_id=portfolio_id)
    logger.info(
        "apply_fills_to_state: portfolio=%s  date=%s  fills=%d  cash=%.2f",
        portfolio_id,
        date,
        len(fills),
        cash_balance,
    )
    return positions
