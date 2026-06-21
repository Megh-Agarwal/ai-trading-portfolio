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
from db.models import Position, PortfolioSnapshot

logger = logging.getLogger(__name__)

_CASH = "CASH"


def get_current_positions(date: str, db: Session) -> dict[str, float]:
    """Return the most recent positions on or before date as {ticker: shares}.

    CASH is included as a ticker with shares == dollars (1 share = $1).
    On the very first call (empty positions table) returns initial_capital as
    CASH plus every universe ticker at 0.0.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.

    Returns:
        Mapping of ticker → shares, including CASH.
    """
    date_obj = datetime.date.fromisoformat(date)

    max_date: datetime.date | None = db.execute(
        select(func.max(Position.date)).where(Position.date <= date_obj)
    ).scalar()

    if max_date is None:
        cfg = load_config("backtest")
        universe = load_config("universe")
        positions: dict[str, float] = {t.ticker: 0.0 for t in universe.tickers}
        positions[_CASH] = float(cfg.initial_capital)
        return positions

    rows = db.execute(
        select(Position).where(Position.date == max_date)
    ).scalars().all()

    return {row.ticker: row.shares for row in rows}


def get_portfolio_value(date: str, db: Session, prices: dict[str, float]) -> float:
    """Compute total portfolio value from current positions and closing prices.

    CASH is always priced at 1.0; any ticker missing from prices also defaults
    to 1.0.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: {ticker: closing_price} for the date.

    Returns:
        Total portfolio value in USD.
    """
    positions = get_current_positions(date, db)
    return sum(shares * prices.get(ticker, 1.0) for ticker, shares in positions.items())


def compute_current_weights(date: str, db: Session, prices: dict[str, float]) -> dict[str, float]:
    """Compute current portfolio weights including CASH.

    Weights sum to 1.0 ± 1e-6 under normal operation. CASH weight is included.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: {ticker: closing_price} for the date.

    Returns:
        Mapping of ticker → weight, including CASH.
    """
    positions = get_current_positions(date, db)
    total = sum(shares * prices.get(ticker, 1.0) for ticker, shares in positions.items())
    if total == 0.0:
        return {ticker: 0.0 for ticker in positions}
    return {
        ticker: shares * prices.get(ticker, 1.0) / total
        for ticker, shares in positions.items()
    }


def write_positions(date: str, positions: dict[str, float], db: Session) -> None:
    """Upsert one row per ticker (including CASH) into the positions table.

    Idempotent: re-running for the same date overwrites rather than duplicates.
    market_value and cost_basis are set to 0.0; the trade execution layer
    (later tickets) is responsible for computing and updating those fields.

    Args:
        date: ISO date string (YYYY-MM-DD).
        positions: {ticker: shares} including CASH.
        db: SQLAlchemy session.
    """
    date_obj = datetime.date.fromisoformat(date)
    for ticker, shares in positions.items():
        db.merge(Position(
            date=date_obj,
            ticker=ticker,
            shares=shares,
            market_value=0.0,
            cost_basis=0.0,
        ))
    db.commit()


def write_portfolio_snapshot(date: str, db: Session, prices: dict[str, float]) -> dict:
    """Compute and persist a portfolio snapshot for date.

    Reads current positions, computes value metrics, writes one row to
    portfolio_snapshot. Idempotent via session.merge on the date primary key.

    Args:
        date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: {ticker: closing_price} for the date.

    Returns:
        Dict with keys total_value, cash, gross_exposure, net_exposure.
    """
    date_obj = datetime.date.fromisoformat(date)
    positions = get_current_positions(date, db)

    cash = positions.get(_CASH, 0.0)  # CASH shares == dollars
    total_value = sum(shares * prices.get(ticker, 1.0) for ticker, shares in positions.items())

    non_cash_values = [
        shares * prices.get(ticker, 1.0)
        for ticker, shares in positions.items()
        if ticker != _CASH
    ]
    gross_exposure = sum(abs(v) for v in non_cash_values)
    net_exposure = sum(non_cash_values)

    snapshot = {
        "total_value": total_value,
        "cash": cash,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
    }

    db.merge(PortfolioSnapshot(
        date=date_obj,
        total_value=total_value,
        cash=cash,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
    ))
    db.commit()

    logger.debug(
        "write_portfolio_snapshot date=%s total=%.2f cash=%.2f gross=%.2f net=%.2f",
        date, total_value, cash, gross_exposure, net_exposure,
    )
    return snapshot
