"""Single-call pipeline that runs all three agents then the aggregator.

Ticket 2.6 — entrypoint for the weekly rebalance sequence.
"""
from __future__ import annotations

import datetime
import logging
import time

from sqlalchemy import delete, func, select
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from agents.macro_agent import MacroAgent
from agents.news_agent import NewsAgent
from agents.polymarket_agent import PolymarketAgent
from aggregator.views import build_views
from config import load_config
from db.models import AgentCall, Signal

logger = logging.getLogger(__name__)

# Agent definitions in execution order.
_AGENT_ORDER = ["sentiment", "macro", "events"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _max_call_id(db: Engine) -> int:
    """Snapshot the current high-water mark in agent_calls before the pipeline runs."""
    with Session(db) as session:
        val = session.execute(select(func.max(AgentCall.call_id))).scalar()
    return val or 0


def _cost_since(call_id_floor: int, db: Engine) -> float:
    """Sum cost_usd for all agent_calls rows created after `call_id_floor`."""
    with Session(db) as session:
        rows = (
            session.execute(
                select(AgentCall).where(AgentCall.call_id > call_id_floor)
            )
            .scalars()
            .all()
        )
    return sum(r.cost_usd or 0.0 for r in rows)


def _write_neutral_signals(agent_name: str, date: datetime.date, db: Engine) -> None:
    """Write zero-value stub rows for a failed agent so build_views can proceed."""
    sectors = [t.ticker for t in load_config("universe").tickers]

    if agent_name == "sentiment":
        rows: list[Signal] = [
            Signal(date=date, agent_name="sentiment", target=s, signal_value=0.0, confidence=0.0)
            for s in sectors
        ]
    elif agent_name == "macro":
        rows = [
            Signal(date=date, agent_name="macro", target="macro_regime", signal_value=0.0, confidence=0.0),
            Signal(date=date, agent_name="macro", target="rate_outlook", signal_value=0.0, confidence=0.0),
        ]
    elif agent_name == "events":
        rows = [
            Signal(date=date, agent_name="events", target=s, signal_value=0.0, confidence=0.0)
            for s in sectors
        ]
    else:
        logger.warning("_write_neutral_signals: unknown agent %r — no stubs written", agent_name)
        return

    with Session(db) as session:
        session.execute(
            delete(Signal).where(Signal.date == date).where(Signal.agent_name == agent_name)
        )
        session.add_all(rows)
        session.commit()

    logger.info("Wrote neutral stubs for failed agent=%s date=%s", agent_name, date)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent_pipeline(
    date: datetime.date,
    db: Engine,
    weights: dict[str, float] | None = None,
    mode: str = "live",
) -> dict:
    """Run all three agents then build_views in sequence for `date`.

    Error handling:
    - If one agent raises, its failure is recorded, neutral (zero) signals are
      written to the DB, and the pipeline continues with the remaining agents.
    - If all three agents fail, the pipeline raises RuntimeError before reaching
      the aggregator.

    Args:
        date: Rebalance date.  All DB signal rows will be for this date.
        db: SQLAlchemy engine.
        weights: Agent weights forwarded to build_views.  None → config defaults.
        mode: "live" or "backtest". Forwarded to build_views for weight selection.
              "backtest" sets polymarket weight to 0% (ADR-012).

    Returns:
        Summary dict with keys:
          date             — the rebalance date (ISO string)
          signals_by_agent — {"sentiment": {"status": "ok"|"error", ...}, ...}
          views            — {"q": list[float], "omega_diag": list[float]}
          total_cost_usd   — sum of cost_usd for all new agent_calls rows
          total_latency_ms — wall-clock time for the full pipeline in milliseconds

    Raises:
        RuntimeError: All three agents failed — no meaningful output is possible.
    """
    logger.info("=== run_agent_pipeline  date=%s ===", date)
    pipeline_start = time.monotonic()
    call_id_floor = _max_call_id(db)

    agents = {
        "sentiment": NewsAgent(),
        "macro": MacroAgent(),
        "events": PolymarketAgent(),
    }

    signals_by_agent: dict[str, dict] = {}
    failed: list[str] = []

    for name in _AGENT_ORDER:
        agent = agents[name]
        t0 = time.monotonic()
        try:
            result = agent.run(date, db)
            elapsed = (time.monotonic() - t0) * 1000
            signals_by_agent[name] = {"status": "ok", "result": result, "latency_ms": elapsed}
            logger.info("Agent %s succeeded  latency=%.0fms", name, elapsed)
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            logger.error("Agent %s failed after %.0fms: %s", name, elapsed, exc, exc_info=True)
            signals_by_agent[name] = {"status": "error", "error": str(exc), "latency_ms": elapsed}
            failed.append(name)
            _write_neutral_signals(name, date, db)

    if len(failed) == len(_AGENT_ORDER):
        raise RuntimeError(
            f"All three agents failed for date={date}. "
            f"Errors: { {k: v['error'] for k, v in signals_by_agent.items()} }"
        )

    if failed:
        logger.warning("%d agent(s) failed — neutral stubs written: %s", len(failed), failed)

    # ---- aggregator -------------------------------------------------------
    q_arr, omega_arr = build_views(date, db, mode=mode, weights=weights)

    total_latency_ms = (time.monotonic() - pipeline_start) * 1000
    total_cost_usd = _cost_since(call_id_floor, db)

    logger.info(
        "Pipeline complete  date=%s  cost=$%.5f  latency=%.0fms  failed=%s",
        date,
        total_cost_usd,
        total_latency_ms,
        failed or "none",
    )

    return {
        "date": date.isoformat(),
        "signals_by_agent": signals_by_agent,
        "views": {
            "q": q_arr.tolist(),
            "omega_diag": omega_arr.diagonal().tolist(),
        },
        "total_cost_usd": total_cost_usd,
        "total_latency_ms": total_latency_ms,
    }
