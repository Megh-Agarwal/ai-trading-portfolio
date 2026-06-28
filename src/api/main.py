"""Read-only portfolio state API — Ticket 6.5.

Run with:
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

Rate limiting is handled at the nginx layer (ticket 6.6) rather than here.
All endpoints are read-only; no auth needed for paper-trading data.
"""

from __future__ import annotations

import datetime
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from db.models import (
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_BACKTEST_SPY,
    PORTFOLIO_LIVE,
    PortfolioSnapshot,
    Position,
    Price,
    RiskEvent,
    Signal,
    TargetWeight,
    Trade,
    View,
)
from eval.metrics import compute_all_metrics

_DB_PATH = Path(
    os.environ.get(
        "DB_PATH",
        str(Path(__file__).parent.parent.parent / "data" / "state.db"),
    )
)
_STALE_THRESHOLD_DAYS = 8  # rebalance runs weekly; >8 days means a missed run

_engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _engine
    _engine = create_engine(
        f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False}
    )
    yield
    if _engine:
        _engine.dispose()


app = FastAPI(
    title="AI Trading Portfolio API",
    description="Read-only API for live portfolio state and performance.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed CORS origin is set via DASHBOARD_ORIGIN env var (configured in .env on EC2).
# Falls back to "*" if unset so local dev and first-run still work.
_CORS_ORIGINS = [o.strip() for o in os.environ.get("DASHBOARD_ORIGIN", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_engine() -> Engine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return _engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/portfolio")
def get_portfolio(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """Current positions, weights, and NAV for the live portfolio."""
    with Session(engine) as session:
        latest_date: datetime.date | None = session.execute(
            select(func.max(PortfolioSnapshot.date)).where(
                PortfolioSnapshot.portfolio_id == PORTFOLIO_LIVE
            )
        ).scalar()

        if latest_date is None:
            raise HTTPException(status_code=404, detail="No portfolio data found")

        snapshot = session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_LIVE)
            .where(PortfolioSnapshot.date == latest_date)
        ).scalar_one()

        positions = session.execute(
            select(Position)
            .where(Position.portfolio_id == PORTFOLIO_LIVE)
            .where(Position.date == latest_date)
            .order_by(Position.ticker)
        ).scalars().all()

        # Fetch the most recent price on or before latest_date for each ticker.
        # positions.market_value is always 0 (write_positions design); compute here.
        tickers = [p.ticker for p in positions if p.ticker != "CASH"]
        prices: dict[str, float] = {}
        for ticker in tickers:
            price_row = session.execute(
                select(Price.adj_close)
                .where(Price.ticker == ticker)
                .where(Price.date <= latest_date)
                .order_by(Price.date.desc())
                .limit(1)
            ).scalar()
            if price_row is not None:
                prices[ticker] = float(price_row)

    total_value = snapshot.total_value
    position_list = [
        {
            "ticker": p.ticker,
            "shares": round(p.shares, 4),
            "market_value": round(p.shares * prices.get(p.ticker, 0.0), 2),
            "weight": round(
                p.shares * prices.get(p.ticker, 0.0) / total_value, 4
            ) if total_value > 0 else 0.0,
        }
        for p in positions
        if p.ticker != "CASH"
    ]

    return {
        "as_of_date": latest_date.isoformat(),
        "nav": round(total_value, 2),
        "cash": round(snapshot.cash, 2),
        "gross_exposure": round(snapshot.gross_exposure, 4),
        "net_exposure": round(snapshot.net_exposure, 4),
        "positions": position_list,
    }


@app.get("/performance")
def get_performance(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """NAV history and performance metrics for the live portfolio."""
    with Session(engine) as session:
        snapshots = session.execute(
            select(PortfolioSnapshot.date, PortfolioSnapshot.total_value)
            .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_LIVE)
            .order_by(PortfolioSnapshot.date)
        ).all()

    if not snapshots:
        raise HTTPException(status_code=404, detail="No performance data found")

    nav_history = [
        {"date": row.date.isoformat(), "nav": round(row.total_value, 2)}
        for row in snapshots
    ]
    start_date = snapshots[0].date
    end_date = snapshots[-1].date

    metrics = compute_all_metrics(
        portfolio_id=PORTFOLIO_LIVE,
        start_date=start_date,
        end_date=end_date,
        db_engine=engine,
        n_bootstrap=200,
    )

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "weeks": len(snapshots),
        "nav_history": nav_history,
        "metrics": metrics,
    }


@app.get("/signals/latest")
def get_latest_signals(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """Most recent week's agent outputs: sentiment per sector, macro regime, Polymarket tilts."""
    with Session(engine) as session:
        latest_date: datetime.date | None = session.execute(
            select(func.max(Signal.date)).where(Signal.portfolio_id == PORTFOLIO_LIVE)
        ).scalar()

        if latest_date is None:
            raise HTTPException(status_code=404, detail="No signal data found")

        signals = session.execute(
            select(Signal)
            .where(Signal.portfolio_id == PORTFOLIO_LIVE)
            .where(Signal.date == latest_date)
            .order_by(Signal.agent_name, Signal.target)
        ).scalars().all()

    sentiment: dict[str, float] = {}
    macro: dict[str, Any] = {}
    events: dict[str, float] = {}

    for s in signals:
        if s.agent_name == "sentiment":
            sentiment[s.target] = round(s.signal_value, 4)
        elif s.agent_name == "macro":
            macro[s.target] = s.signal_value
            if s.confidence is not None:
                macro["confidence"] = round(float(s.confidence), 4)
        elif s.agent_name == "events":
            events[s.target] = round(s.signal_value, 4)

    return {
        "date": latest_date.isoformat(),
        "sentiment": sentiment,
        "macro": macro,
        "events": events,
    }


@app.get("/trades/recent")
def get_recent_trades(
    n: int = Query(default=20, ge=1, le=100),
    engine: Engine = Depends(get_engine),
) -> dict[str, Any]:
    """Last N trades for the live portfolio (default 20, max 100)."""
    with Session(engine) as session:
        trades = session.execute(
            select(Trade)
            .where(Trade.portfolio_id == PORTFOLIO_LIVE)
            .order_by(Trade.date.desc(), Trade.trade_id.desc())
            .limit(n)
        ).scalars().all()

    return {
        "n": len(trades),
        "trades": [
            {
                "trade_id": t.trade_id,
                "date": t.date.isoformat(),
                "ticker": t.ticker,
                "side": t.side,
                "shares": round(t.shares, 4),
                "price": round(t.price, 4),
                "commission": round(t.commission, 4),
            }
            for t in trades
        ],
    }


@app.get("/health")
def get_health(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """Last successful run timestamp and any triggered risk events from that run."""
    with Session(engine) as session:
        last_run_date: datetime.date | None = session.execute(
            select(func.max(PortfolioSnapshot.date)).where(
                PortfolioSnapshot.portfolio_id == PORTFOLIO_LIVE
            )
        ).scalar()

        triggered_risk: list[dict[str, Any]] = []
        if last_run_date is not None:
            risk_rows = session.execute(
                select(RiskEvent)
                .where(RiskEvent.portfolio_id == PORTFOLIO_LIVE)
                .where(RiskEvent.date == last_run_date)
                .where(RiskEvent.triggered == True)  # noqa: E712
                .order_by(RiskEvent.event_id)
            ).scalars().all()
            triggered_risk = [
                {
                    "check_name": r.check_name,
                    "value": r.value,
                    "threshold": r.threshold,
                    "action_taken": r.action_taken,
                }
                for r in risk_rows
            ]

    if last_run_date is None:
        return {
            "status": "never_run",
            "last_run_date": None,
            "days_since_last_run": None,
            "stale_threshold_days": _STALE_THRESHOLD_DAYS,
            "triggered_risk_events": [],
        }

    days_since = (datetime.date.today() - last_run_date).days
    status = "healthy" if days_since <= _STALE_THRESHOLD_DAYS else "stale"

    return {
        "status": status,
        "last_run_date": last_run_date.isoformat(),
        "days_since_last_run": days_since,
        "stale_threshold_days": _STALE_THRESHOLD_DAYS,
        "triggered_risk_events": triggered_risk,
    }


@app.get("/backtest/nav")
def get_backtest_nav(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """NAV history for all four backtest portfolios — used for the comparison chart."""
    _PORTFOLIOS = {
        "LLM (Full)": PORTFOLIO_BACKTEST_FULL,
        "No LLM": PORTFOLIO_BACKTEST_NO_LLM,
        "Equal Weight": PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
        "SPY": PORTFOLIO_BACKTEST_SPY,
    }
    result: dict[str, list[dict[str, Any]]] = {}
    with Session(engine) as session:
        for label, portfolio_id in _PORTFOLIOS.items():
            rows = session.execute(
                select(PortfolioSnapshot.date, PortfolioSnapshot.total_value)
                .where(PortfolioSnapshot.portfolio_id == portfolio_id)
                .order_by(PortfolioSnapshot.date)
            ).all()
            result[label] = [
                {"date": row.date.isoformat(), "nav": round(row.total_value, 2)}
                for row in rows
            ]
    return result


@app.get("/backtest/signals")
def get_backtest_signals(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """All agent signals (sentiment, macro, events) for the full backtest — for heatmaps."""
    with Session(engine) as session:
        rows = session.execute(
            select(Signal.date, Signal.agent_name, Signal.target, Signal.signal_value, Signal.confidence)
            .where(Signal.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .order_by(Signal.date, Signal.agent_name, Signal.target)
        ).all()
    return {
        "signals": [
            {
                "date": row.date.isoformat(),
                "agent": row.agent_name,
                "target": row.target,
                "value": round(row.signal_value, 4),
                "conviction": round(float(row.confidence), 4) if row.confidence is not None else None,
            }
            for row in rows
        ]
    }


@app.get("/backtest/views")
def get_backtest_views(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """Black-Litterman Q vector (expected excess returns) per sector per week."""
    with Session(engine) as session:
        rows = session.execute(
            select(View.date, View.sector, View.expected_return, View.confidence)
            .where(View.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .order_by(View.date, View.sector)
        ).all()
    return {
        "views": [
            {
                "date": row.date.isoformat(),
                "sector": row.sector,
                "expected_return": round(row.expected_return * 100, 4),  # convert to %
                "conviction": round(float(row.confidence), 4) if row.confidence is not None else None,
            }
            for row in rows
        ]
    }


@app.get("/backtest/weights")
def get_backtest_weights(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    """Target weight history for the full LLM backtest portfolio."""
    with Session(engine) as session:
        rows = session.execute(
            select(TargetWeight.date, TargetWeight.sector, TargetWeight.weight)
            .where(TargetWeight.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .order_by(TargetWeight.date, TargetWeight.sector)
        ).all()
    return {
        "weights": [
            {
                "date": row.date.isoformat(),
                "sector": row.sector,
                "weight": round(row.weight, 4),
            }
            for row in rows
        ]
    }
