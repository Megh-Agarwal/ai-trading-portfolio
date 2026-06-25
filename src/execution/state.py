"""Portfolio state read/write functions for the M4 execution layer (Ticket 4.1).

Cash is a first-class position stored as ticker "CASH" where 1 share == $1,
so the whole portfolio (ETF positions + cash) is one consistent dict that
always sums to total portfolio value.
"""

from __future__ import annotations

import datetime
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import load_config
from db.models import PORTFOLIO_LIVE, PortfolioSnapshot, Position

logger = logging.getLogger(__name__)

_CASH = "CASH"


def get_current_positions(
    date: str,
    db: Session,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict[str, float]:
    """Return the most recent positions on or before date as {ticker: shares}.

    CASH is included as a ticker with shares == dollars (1 share = $1).
    On the very first call (empty positions table) returns initial_capital as
    CASH plus every universe ticker at 0.0.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        portfolio_id: Portfolio namespace. Positions are isolated per ID.

    Returns:
        Mapping of ticker → shares, including CASH.
    """
    date_obj = datetime.date.fromisoformat(date)

    max_date: datetime.date | None = db.execute(
        select(func.max(Position.date))
        .where(Position.portfolio_id == portfolio_id)
        .where(Position.date <= date_obj)
    ).scalar()

    if max_date is None:
        cfg = load_config("backtest")
        universe = load_config("universe")
        positions: dict[str, float] = {t.ticker: 0.0 for t in universe.tickers}
        positions[_CASH] = float(cfg.initial_capital)
        return positions

    rows = (
        db.execute(
            select(Position)
            .where(Position.portfolio_id == portfolio_id)
            .where(Position.date == max_date)
        )
        .scalars()
        .all()
    )

    return {row.ticker: row.shares for row in rows}


def get_portfolio_value(
    date: str,
    db: Session,
    prices: dict[str, float],
    portfolio_id: str = PORTFOLIO_LIVE,
) -> float:
    """Compute total portfolio value from current positions and closing prices.

    CASH is always priced at 1.0; any ticker missing from prices also defaults
    to 1.0.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: {ticker: closing_price} for the date.
        portfolio_id: Portfolio namespace.

    Returns:
        Total portfolio value in USD.
    """
    positions = get_current_positions(date, db, portfolio_id=portfolio_id)
    return sum(shares * prices.get(ticker, 1.0) for ticker, shares in positions.items())


def compute_current_weights(
    date: str,
    db: Session,
    prices: dict[str, float],
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict[str, float]:
    """Compute current portfolio weights including CASH.

    Weights sum to 1.0 ± 1e-6 under normal operation. CASH weight is included.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: {ticker: closing_price} for the date.
        portfolio_id: Portfolio namespace.

    Returns:
        Mapping of ticker → weight, including CASH.
    """
    positions = get_current_positions(date, db, portfolio_id=portfolio_id)
    total = sum(shares * prices.get(ticker, 1.0) for ticker, shares in positions.items())
    if total == 0.0:
        return {ticker: 0.0 for ticker in positions}
    return {
        ticker: shares * prices.get(ticker, 1.0) / total for ticker, shares in positions.items()
    }


def write_positions(
    date: str,
    positions: dict[str, float],
    db: Session,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> None:
    """Upsert one row per ticker (including CASH) into the positions table.

    Idempotent: re-running for the same date overwrites rather than duplicates.
    market_value and cost_basis are set to 0.0; the trade execution layer
    (later tickets) is responsible for computing and updating those fields.

    Args:
        date: ISO date string (YYYY-MM-DD).
        positions: {ticker: shares} including CASH.
        db: SQLAlchemy session.
        portfolio_id: Portfolio namespace.
    """
    date_obj = datetime.date.fromisoformat(date)
    for ticker, shares in positions.items():
        db.merge(
            Position(
                portfolio_id=portfolio_id,
                date=date_obj,
                ticker=ticker,
                shares=shares,
                market_value=0.0,
                cost_basis=0.0,
            )
        )
    db.commit()


def write_portfolio_snapshot(
    date: str,
    db: Session,
    prices: dict[str, float],
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict:
    """Compute and persist a portfolio snapshot for date.

    Reads current positions, computes value metrics, writes one row to
    portfolio_snapshot. Idempotent via session.merge on the (portfolio_id, date) PK.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: {ticker: closing_price} for the date.
        portfolio_id: Portfolio namespace.

    Returns:
        Dict with keys total_value, cash, gross_exposure, net_exposure.
    """
    date_obj = datetime.date.fromisoformat(date)
    positions = get_current_positions(date, db, portfolio_id=portfolio_id)

    cash = positions.get(_CASH, 0.0)  # CASH shares == dollars
    total_value = sum(shares * prices.get(ticker, 1.0) for ticker, shares in positions.items())

    non_cash_values = [
        shares * prices.get(ticker, 1.0) for ticker, shares in positions.items() if ticker != _CASH
    ]
    gross_exposure = sum(abs(v) for v in non_cash_values)
    net_exposure = sum(non_cash_values)

    snapshot = {
        "total_value": total_value,
        "cash": cash,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
    }

    db.merge(
        PortfolioSnapshot(
            portfolio_id=portfolio_id,
            date=date_obj,
            total_value=total_value,
            cash=cash,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
        )
    )
    db.commit()

    logger.debug(
        "write_portfolio_snapshot portfolio=%s date=%s total=%.2f cash=%.2f gross=%.2f net=%.2f",
        portfolio_id,
        date,
        total_value,
        cash,
        gross_exposure,
        net_exposure,
    )
    return snapshot
