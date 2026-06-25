"""Performance attribution for the M4 execution layer (Ticket 4.4).

Decomposes portfolio return over a period into per-sector contributions and
cost drag. Used by M5 backtest analysis and M6 dashboard — not on the hot
path for the weekly run.

Sector contribution approximation:
  contribution_i = avg_weight_i × sector_return_i

where avg_weight is the simple average of start-of-period and end-of-period
weights.  The approximation error (vs. continuous daily rebalancing) is
surfaced explicitly in reconcile_attribution's unexplained_pct field.

All return values are decimal fractions (0.05 = 5%), not percentages.
Cost drag is in basis points (1 bp = 0.0001).
"""

from __future__ import annotations

import datetime
import logging

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import PORTFOLIO_LIVE, PortfolioSnapshot, Trade
from execution.state import get_current_positions

logger = logging.getLogger(__name__)

_CASH = "CASH"


def compute_period_return(
    start_date: str,
    end_date: str,
    db: Session,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict:
    """Compute the total portfolio return between two snapshot dates.

    Args:
        start_date: ISO date string (YYYY-MM-DD). Must have a snapshot row.
        end_date: ISO date string (YYYY-MM-DD). Must have a snapshot row.
        db: SQLAlchemy session.

    Returns:
        Dict with total_return_pct (decimal fraction), start_value, end_value.

    Raises:
        ValueError: If either date has no portfolio_snapshot row.
    """
    start_obj = datetime.date.fromisoformat(start_date)
    end_obj = datetime.date.fromisoformat(end_date)

    snap_start = db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date == start_obj)
    ).scalar_one_or_none()
    snap_end = db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date == end_obj)
    ).scalar_one_or_none()

    if snap_start is None:
        raise ValueError(f"No portfolio_snapshot row for start_date={start_date}")
    if snap_end is None:
        raise ValueError(f"No portfolio_snapshot row for end_date={end_date}")

    start_value = snap_start.total_value
    end_value = snap_end.total_value
    total_return_pct = (end_value - start_value) / start_value

    return {
        "total_return_pct": total_return_pct,
        "start_value": start_value,
        "end_value": end_value,
    }


def compute_sector_contribution(
    start_date: str,
    end_date: str,
    db: Session,
    prices: pd.DataFrame,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict[str, float]:
    """Compute per-sector return contributions over the period.

    For each non-CASH ticker: contribution = avg_weight × sector_return.
    CASH is always assigned contribution 0.0 (no yield in this model).

    prices must have tickers as columns and date strings (YYYY-MM-DD) or
    pandas Timestamps as the index.

    Args:
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.
        prices: DataFrame of closing prices; columns = tickers, index = dates.

    Returns:
        {ticker: contribution_fraction} including CASH (always 0.0).

    Raises:
        ValueError: If snapshot rows are missing for either boundary date.
    """
    start_obj = datetime.date.fromisoformat(start_date)
    end_obj = datetime.date.fromisoformat(end_date)

    snap_start = db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date == start_obj)
    ).scalar_one_or_none()
    snap_end = db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date == end_obj)
    ).scalar_one_or_none()

    if snap_start is None:
        raise ValueError(f"No portfolio_snapshot row for start_date={start_date}")
    if snap_end is None:
        raise ValueError(f"No portfolio_snapshot row for end_date={end_date}")

    pos_start = get_current_positions(start_date, db, portfolio_id=portfolio_id)
    pos_end = get_current_positions(end_date, db, portfolio_id=portfolio_id)

    all_tickers = (set(pos_start.keys()) | set(pos_end.keys())) - {_CASH}
    contributions: dict[str, float] = {}

    for ticker in sorted(all_tickers):
        if ticker not in prices.columns:
            logger.warning("compute_sector_contribution: %s not in prices, skipping", ticker)
            continue

        try:
            price_start = float(prices.loc[start_date, ticker])
            price_end = float(prices.loc[end_date, ticker])
        except KeyError:
            logger.warning(
                "compute_sector_contribution: missing price row for %s on %s or %s",
                ticker,
                start_date,
                end_date,
            )
            continue

        if price_start <= 0:
            logger.warning("compute_sector_contribution: non-positive start price for %s", ticker)
            continue

        val_start = pos_start.get(ticker, 0.0) * price_start
        val_end = pos_end.get(ticker, 0.0) * price_end

        w_start = val_start / snap_start.total_value if snap_start.total_value > 0 else 0.0
        w_end = val_end / snap_end.total_value if snap_end.total_value > 0 else 0.0
        avg_weight = (w_start + w_end) / 2.0

        sector_return = (price_end - price_start) / price_start
        contributions[ticker] = avg_weight * sector_return

    contributions[_CASH] = 0.0  # CASH earns 0% in this model
    return contributions


def compute_cost_drag(
    start_date: str,
    end_date: str,
    db: Session,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict:
    """Compute total transaction costs and express as portfolio basis points.

    Sums commission from all trades in [start_date, end_date] (inclusive).
    Basis points are computed against the simple average of start and end
    portfolio values from portfolio_snapshot.

    Args:
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string (YYYY-MM-DD).
        db: SQLAlchemy session.

    Returns:
        Dict with total_cost_usd and cost_drag_bps.
    """
    start_obj = datetime.date.fromisoformat(start_date)
    end_obj = datetime.date.fromisoformat(end_date)

    total_cost_usd: float = (
        db.execute(
            select(func.sum(Trade.commission)).where(
                Trade.portfolio_id == portfolio_id,
                Trade.date >= start_obj,
                Trade.date <= end_obj,
            )
        ).scalar()
        or 0.0
    )

    snap_start = db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date == start_obj)
    ).scalar_one_or_none()
    snap_end = db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date == end_obj)
    ).scalar_one_or_none()

    values = [s.total_value for s in (snap_start, snap_end) if s is not None]
    avg_value = sum(values) / len(values) if values else 0.0

    cost_drag_bps = (total_cost_usd / avg_value * 10_000) if avg_value > 0 else 0.0

    return {"total_cost_usd": total_cost_usd, "cost_drag_bps": cost_drag_bps}


def reconcile_attribution(
    total_return: float,
    sector_contributions: dict[str, float],
    cost_drag_bps: float,
) -> dict:
    """Check whether sector contributions reconcile to total return minus cost drag.

    explained = sum(contributions) - cost_drag_fraction
    unexplained_pct = total_return - explained

    The unexplained gap comes from the weight-averaging approximation and is
    expected to be small (< 0.5% absolute) for weekly rebalances. It is
    always surfaced explicitly — never silently ignored.

    Args:
        total_return: Portfolio return as decimal fraction (e.g. 0.05 = 5%).
        sector_contributions: {ticker: contribution_fraction} from
            compute_sector_contribution.
        cost_drag_bps: Cost drag in basis points from compute_cost_drag.

    Returns:
        Dict with sum_contributions, cost_drag_fraction, explained,
        unexplained_pct, and the original total_return.
    """
    sum_contributions = sum(sector_contributions.values())
    cost_drag_fraction = cost_drag_bps / 10_000
    explained = sum_contributions - cost_drag_fraction
    unexplained_pct = total_return - explained

    logger.debug(
        "reconcile: total=%.6f  sum_contrib=%.6f  cost_drag=%.4fbps  "
        "explained=%.6f  unexplained=%.6f",
        total_return,
        sum_contributions,
        cost_drag_bps,
        explained,
        unexplained_pct,
    )

    return {
        "total_return": total_return,
        "sum_contributions": sum_contributions,
        "cost_drag_fraction": cost_drag_fraction,
        "explained": explained,
        "unexplained_pct": unexplained_pct,
    }
