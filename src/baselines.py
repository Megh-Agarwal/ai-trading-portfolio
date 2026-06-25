"""SPY buy-and-hold benchmark for backtesting — Ticket 5.3.

Buys SPY with full initial_capital on week 1, holds unchanged for all
subsequent weeks. Portfolio value is tracked by re-reading the SPY close
price each week; no new trades are generated after the initial buy.
"""

from __future__ import annotations

import datetime
import logging
import math

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from config import load_config
from db.models import PORTFOLIO_BACKTEST_SPY, Price, TargetWeight, Trade
from execution.state import get_current_positions, write_portfolio_snapshot, write_positions

logger = logging.getLogger(__name__)

_SPY = "SPY"
_CASH = "CASH"


def _fetch_spy_close(date_obj: datetime.date, db_engine) -> float | None:
    """Return the most recent SPY adj_close on or before date_obj."""
    with Session(db_engine) as session:
        row = session.execute(
            select(Price.adj_close)
            .where(and_(Price.ticker == _SPY, Price.date <= date_obj))
            .order_by(Price.date.desc())
            .limit(1)
        ).scalar()
    return float(row) if row is not None else None


def run_spy_benchmark(
    date_str: str,
    db_engine,
    mode: str = "backtest",
    portfolio_id: str = PORTFOLIO_BACKTEST_SPY,
) -> dict:
    """Run one week of the SPY buy-and-hold benchmark.

    Week 1: buy SPY shares with full initial_capital (writes one Trade).
    Weeks 2–N: hold unchanged — carry positions forward, update snapshot
    using the current SPY close. No new trades after week 1.

    Args:
        date_str: ISO date string (YYYY-MM-DD).
        db_engine: SQLAlchemy Engine connected to state.db.
        mode: Passed through to the result dict.
        portfolio_id: Portfolio namespace for DB isolation.

    Returns:
        Summary dict with the same contract as run_weekly, so callers need
        no special-casing on the return value.

    Raises:
        RuntimeError: No SPY price in DB for the given date.
    """
    date_obj = datetime.date.fromisoformat(date_str)

    spy_price = _fetch_spy_close(date_obj, db_engine)
    if spy_price is None:
        raise RuntimeError(f"No SPY price in DB for date={date_str}")

    cfg_opt = load_config("optimizer")
    tc = cfg_opt.transaction_costs
    cost_rate = (tc.spread_bps + tc.slippage_bps) / 10_000.0

    with Session(db_engine) as session:
        current_positions = get_current_positions(date_str, session, portfolio_id=portfolio_id)

    is_first_run = current_positions.get(_SPY, 0.0) == 0.0

    if is_first_run:
        initial_capital = float(load_config("backtest").initial_capital)
        # Floor shares so gross + cost never exceeds initial_capital
        spy_shares = math.floor(initial_capital / (spy_price * (1.0 + cost_rate)))
        gross_value = spy_shares * spy_price
        cost_usd = gross_value * cost_rate
        cash_remaining = initial_capital - gross_value - cost_usd

        with Session(db_engine) as session:
            session.add(
                Trade(
                    portfolio_id=portfolio_id,
                    date=date_obj,
                    ticker=_SPY,
                    side="buy",
                    shares=float(spy_shares),
                    price=spy_price,
                    commission=cost_usd,
                    slippage=0.0,
                )
            )
            session.commit()

        new_positions: dict[str, float] = {_SPY: float(spy_shares), _CASH: cash_remaining}
        logger.info(
            "SPY first buy  date=%s  shares=%d  price=%.2f  cost=$%.2f  cash=$%.2f",
            date_obj,
            spy_shares,
            spy_price,
            cost_usd,
            cash_remaining,
        )
    else:
        # Hold: carry the existing (week-1) positions forward to today's date.
        new_positions = dict(current_positions)

    prices = {_SPY: spy_price, _CASH: 1.0}

    with Session(db_engine) as session:
        write_positions(date_str, new_positions, session, portfolio_id=portfolio_id)
        session.merge(
            TargetWeight(portfolio_id=portfolio_id, date=date_obj, sector=_SPY, weight=1.0)
        )
        session.commit()
        snapshot = write_portfolio_snapshot(date_str, session, prices, portfolio_id=portfolio_id)

    spy_shares_held = new_positions.get(_SPY, 0.0)
    spy_value = spy_shares_held * spy_price
    total_value = snapshot["total_value"]

    logger.info(
        "SPY benchmark  date=%s  price=%.2f  shares=%.0f  spy_value=$%.2f  total=$%.2f",
        date_obj,
        spy_price,
        spy_shares_held,
        spy_value,
        total_value,
    )

    return {
        "date": date_str,
        "mode": mode,
        "skipped": False,
        "portfolio_id": portfolio_id,
        "llm_cost_usd": 0.0,
        "weights_after": {_SPY: 1.0},
        "ending_portfolio_value": total_value,
        "spy_price": spy_price,
        "spy_shares": spy_shares_held,
    }
