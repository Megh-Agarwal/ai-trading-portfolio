"""Weekly rebalance core logic — Ticket 4.5.

This module is the importable core; tests use it directly.
The CLI entry point is scripts/run_weekly.py which handles arg-parsing
and sys.exit codes.

Sequence for each run:
  1. Idempotency check
  2. Ingest fresh data (prices, macro, news, Polymarket if live)
  3. Agent pipeline  → signals + views
  4. Optimization pipeline → target weights + risk checks
  5. Load current positions + most-recent prices
  6. Generate orders (sells first)
  7. Simulate fills → write trades
  8. Apply fills to state → write positions + snapshot
  9. Return structured summary
"""

from __future__ import annotations

import datetime
import logging
import traceback

from sqlalchemy import Engine, and_, func, select
from sqlalchemy.orm import Session

from agents.pipeline import run_agent_pipeline
from config import load_config
from db.models import (
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_LIVE,
    Position,
    Price,
    TargetWeight,
)
from execution.fill_simulator import apply_fills_to_state, simulate_all_fills
from execution.orders import generate_orders, validate_orders_affordable
from execution.state import (
    compute_current_weights,
    get_current_positions,
    get_portfolio_value,
    write_portfolio_snapshot,
)
from optimizer.pipeline import run_equal_weight_pipeline, run_optimization_pipeline

logger = logging.getLogger(__name__)

_PRICE_LOOKBACK_DAYS = 14  # generous window for weekends/holidays
_NEWS_LOOKBACK_DAYS = 7


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ingest_fresh_data(date_obj: datetime.date, mode: str, db_engine: Engine) -> None:
    """Ingest prices, macro, news (and Polymarket in live mode) for the past week.

    Price failure is fatal — every other step depends on current prices.
    Macro, news, and Polymarket failures are non-fatal; logged as warnings.
    """
    from ingestion.holdings import load_holdings
    from ingestion.macro import SERIES_IDS, fetch_macro, write_macro
    from ingestion.news import fetch_company_news, write_news
    from ingestion.prices import fetch_prices, write_prices

    universe = load_config("universe")
    tickers = [t.ticker for t in universe.tickers] + [universe.benchmark]
    price_start = date_obj - datetime.timedelta(days=_PRICE_LOOKBACK_DAYS)
    news_start = date_obj - datetime.timedelta(days=_NEWS_LOOKBACK_DAYS)

    # Prices — critical; re-raise on failure
    try:
        df = fetch_prices(tickers, price_start, date_obj)
        write_prices(df, db_engine)
        logger.info("Prices ingested  %s → %s  (%d tickers)", price_start, date_obj, len(tickers))
    except Exception:
        logger.critical("Price ingestion failed — aborting\n%s", traceback.format_exc())
        raise

    # Macro — non-fatal
    try:
        df_macro = fetch_macro(SERIES_IDS, price_start, date_obj)
        write_macro(df_macro, db_engine)
        logger.info("Macro ingested (%d series)", len(SERIES_IDS))
    except Exception:
        logger.warning("Macro ingestion failed (non-fatal)\n%s", traceback.format_exc())

    # News — non-fatal; per-ticker failures silently skipped
    try:
        holdings = load_holdings()
        n_articles = 0
        for etf, comp_tickers in holdings.items():
            for ticker in comp_tickers:
                try:
                    articles = fetch_company_news(ticker, news_start, date_obj)
                    n_articles += write_news(articles, sector=etf, engine=db_engine)
                except Exception:
                    logger.debug("News fetch skipped for %s", ticker)
        logger.info("News ingested  %d new articles", n_articles)
    except Exception:
        logger.warning("News ingestion failed (non-fatal)\n%s", traceback.format_exc())

    # Polymarket — live mode only, non-fatal
    if mode == "live":
        try:
            from ingestion.polymarket import (
                fetch_current_state,
                load_curated_markets,
                write_polymarket,
            )

            markets = load_curated_markets()
            snapshots = fetch_current_state(markets)
            write_polymarket(snapshots, db_engine)
            logger.info("Polymarket ingested  %d markets", len(markets))
        except Exception:
            logger.warning("Polymarket ingestion failed (non-fatal)\n%s", traceback.format_exc())


def _fetch_prices_for_date(
    date_obj: datetime.date,
    tickers: list[str],
    db_engine: Engine,
) -> dict[str, float]:
    """Return the most recent adj_close on or before date_obj for each ticker."""
    with Session(db_engine) as session:
        subq = (
            select(Price.ticker, func.max(Price.date).label("max_date"))
            .where(and_(Price.date <= date_obj, Price.ticker.in_(tickers)))
            .group_by(Price.ticker)
            .subquery()
        )
        rows = session.execute(
            select(Price.ticker, Price.adj_close).join(
                subq,
                and_(Price.ticker == subq.c.ticker, Price.date == subq.c.max_date),
            )
        ).all()

    prices = {row.ticker: float(row.adj_close) for row in rows if row.adj_close is not None}
    missing = [t for t in tickers if t not in prices]
    if missing:
        logger.warning("No price data for %d tickers: %s", len(missing), missing)
    return prices


def _already_executed(
    date_obj: datetime.date,
    db_engine: Engine,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> bool:
    """Return True if target_weights and positions both exist for this date and portfolio_id."""
    with Session(db_engine) as session:
        has_weights = (
            session.execute(
                select(func.count())
                .select_from(TargetWeight)
                .where(TargetWeight.portfolio_id == portfolio_id)
                .where(TargetWeight.date == date_obj)
            ).scalar()
            or 0
        ) > 0
        has_positions = (
            session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == portfolio_id)
                .where(Position.date == date_obj)
            ).scalar()
            or 0
        ) > 0

    return has_weights and has_positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_weekly(
    date_str: str,
    mode: str,
    db_engine: Engine,
    force: bool = False,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict:
    """Run one full weekly rebalance cycle.

    Args:
        date_str: ISO date string (YYYY-MM-DD) for the rebalance.
        mode: "backtest" or "live". Controls Polymarket aggregator weight (ADR-012).
        db_engine: SQLAlchemy Engine for state.db.
        force: Override the idempotency check and re-run even if already executed.
        portfolio_id: Portfolio namespace. All writes (signals, views, weights,
            trades, positions, snapshots, risk_events) are scoped to this ID.
            Defaults to "live" for backward compatibility.

    Returns:
        Summary dict. On a skipped run: {"skipped": True, "reason": ..., "date": ...}.

    Raises:
        Exception: Any unhandled failure from optimization or execution steps.
            Ingestion failures are non-fatal (except prices).
    """
    date_obj = datetime.date.fromisoformat(date_str)
    logger.info(
        "=== run_weekly  date=%s  mode=%s  portfolio=%s  force=%s ===",
        date_str,
        mode,
        portfolio_id,
        force,
    )

    # ── idempotency guard ─────────────────────────────────────────────────
    if not force and _already_executed(date_obj, db_engine, portfolio_id=portfolio_id):
        msg = f"Rebalance for {date_str} portfolio={portfolio_id} already executed, skipping"
        logger.info(msg)
        return {"skipped": True, "reason": msg, "date": date_str}

    cfg = load_config("optimizer")
    tickers = load_config("universe").ticker_list
    summary: dict = {"date": date_str, "mode": mode, "skipped": False, "portfolio_id": portfolio_id}

    # ── step 1: ingest ────────────────────────────────────────────────────
    _ingest_fresh_data(date_obj, mode, db_engine)

    # ── step 2: agent pipeline ────────────────────────────────────────────
    # Baseline portfolios (no-LLM, equal-weight) skip the agent pipeline entirely.
    _baseline_ids = (PORTFOLIO_BACKTEST_NO_LLM, PORTFOLIO_BACKTEST_EQUAL_WEIGHT)
    if portfolio_id not in _baseline_ids:
        agent_result = run_agent_pipeline(date_obj, db_engine, mode=mode, portfolio_id=portfolio_id)
        summary["llm_cost_usd"] = agent_result["total_cost_usd"]
        logger.info("Agent pipeline complete  cost=$%.5f", agent_result["total_cost_usd"])
    else:
        summary["llm_cost_usd"] = 0.0
        logger.info("Skipping agent pipeline for baseline portfolio=%s", portfolio_id)

    # ── step 3: optimization pipeline ─────────────────────────────────────
    if portfolio_id == PORTFOLIO_BACKTEST_EQUAL_WEIGHT:
        opt_result = run_equal_weight_pipeline(date_str, db_engine, portfolio_id=portfolio_id)
    elif portfolio_id == PORTFOLIO_BACKTEST_NO_LLM:
        opt_result = run_optimization_pipeline(
            date_str, db_engine, mode=mode, portfolio_id=portfolio_id, force_zero_views=True
        )
    else:
        opt_result = run_optimization_pipeline(
            date_str, db_engine, mode=mode, portfolio_id=portfolio_id
        )
    target_weights: dict[str, float] = opt_result["weights"]
    summary.update(
        {
            "vol_constraint_status": opt_result["vol_constraint_status"],
            "any_risk_triggered": opt_result["any_risk_triggered"],
            "estimated_transaction_cost_usd": opt_result["estimated_cost_usd"],
            "turnover": opt_result["turnover"],
        }
    )

    # ── step 4: load state ────────────────────────────────────────────────
    prices = _fetch_prices_for_date(date_obj, tickers, db_engine)

    with Session(db_engine) as session:
        current_positions = get_current_positions(date_str, session, portfolio_id=portfolio_id)
        weights_before = compute_current_weights(
            date_str, session, prices, portfolio_id=portfolio_id
        )
        portfolio_value_before = get_portfolio_value(
            date_str, session, prices, portfolio_id=portfolio_id
        )

    summary["portfolio_value_before"] = portfolio_value_before
    summary["weights_before"] = {t: weights_before.get(t, 0.0) for t in tickers}
    logger.info("Portfolio value before rebalance: $%.2f", portfolio_value_before)

    # ── step 5: generate orders ───────────────────────────────────────────
    orders = generate_orders(
        target_weights=target_weights,
        current_positions=current_positions,
        portfolio_value=portfolio_value_before,
        prices=prices,
        min_trade_threshold=cfg.transaction_costs.min_trade_threshold,
    )
    orders = [o for o in orders if o.side == "sell"] + [o for o in orders if o.side == "buy"]
    summary["orders_count"] = len(orders)

    available_cash = current_positions.get("CASH", 0.0)
    tc = cfg.transaction_costs
    cost_rate = (tc.spread_bps + tc.slippage_bps) / 10_000.0
    orders = validate_orders_affordable(orders, available_cash, cost_rate=cost_rate)

    # ── steps 6-8: fills + state update + snapshot ─────────────────────────
    with Session(db_engine) as session:
        fills = simulate_all_fills(
            orders, date_str, session, cfg.transaction_costs, portfolio_id=portfolio_id
        )
        apply_fills_to_state(date_str, fills, current_positions, session, portfolio_id=portfolio_id)
        snapshot = write_portfolio_snapshot(date_str, session, prices, portfolio_id=portfolio_id)

    actual_cost_usd = sum(f["cost_usd"] for f in fills)
    summary["actual_transaction_cost_usd"] = actual_cost_usd
    summary["ending_portfolio_value"] = snapshot["total_value"]
    summary["weights_after"] = target_weights

    # ── step 9: risk events triggered this run ─────────────────────────────
    triggered = [r for r in opt_result.get("risk_checks", []) if not r.passed]
    summary["risk_events_triggered"] = len(triggered)

    logger.info(
        "=== Rebalance complete  portfolio=%s  date=%s  orders=%d  cost=$%.2f  portfolio=$%.2f ===",
        portfolio_id,
        date_str,
        len(orders),
        actual_cost_usd,
        snapshot["total_value"],
    )
    return summary
