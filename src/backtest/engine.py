"""Backtest engine — Ticket 5.4.

Iterates run_weekly over all Fridays in [start_date, end_date] for all four
backtest portfolio IDs. Idempotent: completed (date, portfolio_id) pairs are
skipped automatically via run_weekly's existing _already_executed guard.

Ordering per date: PORTFOLIO_BACKTEST_FULL runs first so its LLM calls are
cached before the other three portfolios (which make no new agent calls).
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from db.models import (
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_BACKTEST_SPY,
)
from weekly_run import run_weekly

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# Run full portfolio first per date: its LLM cache is then warm for subsequent portfolios.
_PORTFOLIO_ORDER = [
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_SPY,
]


def _generate_fridays(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Return all Fridays (weekday=4) in [start, end] inclusive."""
    days_ahead = (4 - start.weekday()) % 7
    current = start + datetime.timedelta(days=days_ahead)
    fridays: list[datetime.date] = []
    while current <= end:
        fridays.append(current)
        current += datetime.timedelta(weeks=1)
    return fridays


def run_backtest(
    start_date: str | datetime.date,
    end_date: str | datetime.date,
    db_engine: Engine,
) -> dict:
    """Run the full backtest for all four portfolio IDs over every Friday in range.

    For each Friday: PORTFOLIO_BACKTEST_FULL executes first (real agent calls →
    cached), then PORTFOLIO_BACKTEST_NO_LLM, PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    and PORTFOLIO_BACKTEST_SPY (no agent calls, cache shared). All four use
    mode="backtest" and skip_ingest=True — price/macro/news data must be
    pre-loaded in the DB before calling this function.

    Idempotent and resumable: if interrupted, a re-run will skip (date,
    portfolio_id) pairs already committed to the DB and complete the rest.

    Args:
        start_date: First rebalance date (advances to the next Friday if not one).
        end_date: Last rebalance date (inclusive).
        db_engine: SQLAlchemy Engine connected to state.db.

    Returns:
        Summary dict with keys:
            start_date, end_date, weeks_total, portfolio_weeks_completed,
            portfolio_weeks_skipped, total_llm_cost_usd.
    """
    start = datetime.date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = datetime.date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    fridays = _generate_fridays(start, end)
    total_weeks = len(fridays)

    portfolio_completed: dict[str, int] = {pid: 0 for pid in _PORTFOLIO_ORDER}
    portfolio_skipped: dict[str, int] = {pid: 0 for pid in _PORTFOLIO_ORDER}
    total_cost = 0.0

    logger.info(
        "=== run_backtest START  start=%s  end=%s  fridays=%d ===",
        start,
        end,
        total_weeks,
    )

    for week_num, friday in enumerate(fridays, start=1):
        date_str = str(friday)
        logger.info("--- Week %d/%d  date=%s ---", week_num, total_weeks, date_str)

        for portfolio_id in _PORTFOLIO_ORDER:
            result = run_weekly(
                date_str=date_str,
                mode="backtest",
                db_engine=db_engine,
                skip_ingest=True,
                portfolio_id=portfolio_id,
            )

            if result.get("skipped"):
                portfolio_skipped[portfolio_id] += 1
                logger.info(
                    "  SKIP  portfolio=%s  reason=%s",
                    portfolio_id,
                    result.get("reason", ""),
                )
            else:
                portfolio_completed[portfolio_id] += 1
                week_cost = result.get("llm_cost_usd", 0.0)
                total_cost += week_cost
                logger.info(
                    "  OK    portfolio=%-30s  cost=$%.5f  value=$%.2f  cumulative_cost=$%.5f",
                    portfolio_id,
                    week_cost,
                    result.get("ending_portfolio_value", 0.0),
                    total_cost,
                )

    logger.info(
        "=== run_backtest COMPLETE  weeks=%d  total_llm_cost=$%.5f ===",
        total_weeks,
        total_cost,
    )
    logger.info(
        "  completed: %s",
        {pid: portfolio_completed[pid] for pid in _PORTFOLIO_ORDER},
    )
    logger.info(
        "  skipped:   %s",
        {pid: portfolio_skipped[pid] for pid in _PORTFOLIO_ORDER},
    )

    return {
        "start_date": str(start),
        "end_date": str(end),
        "weeks_total": total_weeks,
        "portfolio_weeks_completed": portfolio_completed,
        "portfolio_weeks_skipped": portfolio_skipped,
        "total_llm_cost_usd": total_cost,
    }
