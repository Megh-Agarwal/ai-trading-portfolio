"""Views builder — merges agent signals into Black-Litterman (Q, Omega).

All tunable parameters (max_excess_return_annual, omega_base, regime_scale_intercept,
regime_scale_slope, agent weights) are read from config/optimizer.yaml (ADR-011, ADR-012).
"""
from __future__ import annotations

import datetime
import logging

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from config import load_config
from db.models import Signal, View

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Non-hyperparameter constants — these are implementation details,
# not backtest knobs. They do NOT live in config.
# ------------------------------------------------------------------
_WEEKS_PER_YEAR: int = 52
_MIN_CONVICTION: float = 0.01  # floor preventing division-by-zero in Omega

_SENTIMENT_AGENT = "sentiment"
_MACRO_AGENT = "macro"
_EVENTS_AGENT = "events"
_MACRO_REGIME_TARGET = "macro_regime"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _fetch_signals(date: datetime.date, db: Engine) -> dict[str, list[dict]]:
    """Return all signal rows for `date`, grouped by agent_name."""
    with Session(db) as session:
        rows = session.execute(select(Signal).where(Signal.date == date)).scalars().all()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.agent_name, []).append(
            {
                "target": row.target,
                "signal_value": row.signal_value,
                "confidence": row.confidence if row.confidence is not None else 0.0,
            }
        )
    return grouped


def _parse_macro(signals_by_agent: dict[str, list[dict]]) -> tuple[float, float]:
    """Return (regime_float, macro_confidence).  Defaults to neutral + 0 confidence."""
    for row in signals_by_agent.get(_MACRO_AGENT, []):
        if row["target"] == _MACRO_REGIME_TARGET:
            return row["signal_value"], row["confidence"]
    return 0.0, 0.0


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_views(
    date: datetime.date,
    db: Engine,
    mode: str = "live",
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge agent signals into Black-Litterman views for `date`.

    All hyperparameters are read from config/optimizer.yaml at call time.

    Args:
        date: Rebalance date. Agent signals for this date must exist in the DB.
        db: SQLAlchemy engine.
        mode: "live" or "backtest". Selects the aggregator_weights config stanza
              when weights is None. "backtest" sets polymarket weight to 0.0 because
              polymarket_raw has no historical data (ADR-012).
        weights: Override agent weight dict with keys "news", "macro", "polymarket".
                 Must sum to ≤ 1.0. Overrides mode-based config weights.

    Returns:
        Q: np.ndarray of shape (N,) — weekly expected excess returns per sector.
        Omega: np.ndarray of shape (N, N) — diagonal view-uncertainty matrix.

    Raises:
        ValueError: No signal rows exist in the DB for this date, or weights sum > 1.0.
    """
    cfg = load_config("optimizer")
    agg = cfg.aggregator
    max_return_weekly = agg.max_excess_return_annual / _WEEKS_PER_YEAR
    omega_base = agg.omega_base
    rm_intercept = agg.regime_scale_intercept
    rm_slope = agg.regime_scale_slope

    if weights is None:
        w_cfg = cfg.aggregator_weights
        w = w_cfg.backtest if mode == "backtest" else w_cfg.live
        weights = {"news": w.news, "macro": w.macro, "polymarket": w.polymarket}

    weight_sum = sum(weights.values())
    if weight_sum > 1.0 + 1e-6:
        raise ValueError(
            f"Agent weights must sum to ≤ 1.0, got {weight_sum:.4f}. "
            f"Weights: {weights}"
        )

    sectors = [t.ticker for t in load_config("universe").tickers]
    n = len(sectors)

    signals_by_agent = _fetch_signals(date, db)
    if not signals_by_agent:
        raise ValueError(
            f"No signal rows found for date={date}. Run all three agents first."
        )

    # ----------------------------------------------------------------
    # Macro regime
    # ----------------------------------------------------------------
    macro_regime, macro_conf = _parse_macro(signals_by_agent)
    regime_scale = rm_intercept + rm_slope * macro_regime  # ∈ [0.50, 1.00]

    # ----------------------------------------------------------------
    # Index per-sector signals by ticker
    # ----------------------------------------------------------------
    sentiment_by_sector: dict[str, dict] = {
        r["target"]: r for r in signals_by_agent.get(_SENTIMENT_AGENT, [])
    }
    events_by_sector: dict[str, dict] = {
        r["target"]: r for r in signals_by_agent.get(_EVENTS_AGENT, [])
    }

    # ----------------------------------------------------------------
    # Build Q and Omega row by row
    # ----------------------------------------------------------------
    q_values: list[float] = []
    omega_diag: list[float] = []
    view_rows: list[View] = []

    for sector in sectors:
        news_row = sentiment_by_sector.get(sector)
        poly_row = events_by_sector.get(sector)

        news_signal = news_row["signal_value"] if news_row else 0.0
        news_conv = news_row["confidence"] if news_row else 0.0
        poly_signal = poly_row["signal_value"] if poly_row else 0.0
        poly_conf = poly_row["confidence"] if poly_row else 0.0

        raw_signal = (
            weights["news"] * news_signal * news_conv
            + weights["polymarket"] * poly_signal * poly_conf
        )
        scaled_signal = raw_signal * regime_scale

        q = scaled_signal * max_return_weekly
        q_values.append(q)

        agg_conviction = (
            weights["news"] * news_conv
            + weights["macro"] * macro_conf
            + weights["polymarket"] * poly_conf
        ) * regime_scale

        omega_entry = omega_base / max(agg_conviction, _MIN_CONVICTION)
        omega_diag.append(omega_entry)

        view_rows.append(
            View(
                date=date,
                sector=sector,
                expected_return=q,
                confidence=float(agg_conviction),
            )
        )

    q_arr = np.array(q_values, dtype=float)
    omega_arr = np.diag(omega_diag)

    # ----------------------------------------------------------------
    # Write to DB (idempotent: delete-before-insert)
    # ----------------------------------------------------------------
    with Session(db) as session:
        session.execute(delete(View).where(View.date == date))
        session.add_all(view_rows)
        session.commit()

    logger.info(
        "build_views date=%s  mode=%s  |Q|_max=%.5f  regime=%.1f  scale=%.2f  sectors=%d",
        date,
        mode,
        float(np.abs(q_arr).max()) if q_arr.size else 0.0,
        macro_regime,
        regime_scale,
        n,
    )

    return q_arr, omega_arr
