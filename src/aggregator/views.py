"""Views builder — merges agent signals into Black-Litterman (Q, Omega)."""
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
# Tunable parameters — see decisions.md ADR-011
# ------------------------------------------------------------------
_MAX_EXCESS_RETURN_ANNUAL: float = 0.05   # 5 % annualised at a ±1 signal
_WEEKS_PER_YEAR: int = 52
_MAX_EXCESS_RETURN_WEEKLY: float = _MAX_EXCESS_RETURN_ANNUAL / _WEEKS_PER_YEAR

# Omega: view variance at unit conviction.  Inverse-proportional scaling means
# a fully-confident view (conviction=1) gets omega=_OMEGA_BASE; a zero-confidence
# view gets omega=_OMEGA_BASE / _MIN_CONVICTION (large uncertainty → tiny BL weight).
_OMEGA_BASE: float = 0.0001   # ≈ 1 bp² weekly variance at max conviction
_MIN_CONVICTION: float = 0.01  # floor preventing division-by-zero

# Macro regime scaling: risk_off(-1)→0.50, neutral(0)→0.75, risk_on(+1)→1.00
# Linear: scale = _RM_INTERCEPT + _RM_SLOPE * regime_float
_RM_INTERCEPT: float = 0.75
_RM_SLOPE: float = 0.25

_DEFAULT_WEIGHTS: dict[str, float] = {"news": 0.4, "macro": 0.3, "polymarket": 0.3}

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
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge agent signals into Black-Litterman views for `date`.

    Requires that all three agents have already written their signals to the DB
    for this date.  The function is idempotent — re-running overwrites the views
    table rows for `date`.

    Algorithm (per sector):
    1. Directional signal  = weighted sum of (news_signal × news_conviction,
                                               poly_signal × poly_confidence).
       Macro does not supply per-sector direction; its regime scales everything.
    2. Regime scale        = 0.75 + 0.25 × macro_regime_float
                             → 0.50 (risk_off), 0.75 (neutral), 1.00 (risk_on)
    3. Q_sector            = directional_signal × regime_scale × MAX_WEEKLY_RETURN
    4. Aggregate conviction = (w_news × news_conv + w_macro × macro_conf
                               + w_poly × poly_conf) × regime_scale
    5. Omega_sector        = OMEGA_BASE / max(aggregate_conviction, MIN_CONVICTION)

    Args:
        date: Rebalance date.  Agent signals for this date must exist in the DB.
        db: SQLAlchemy engine.
        weights: Agent weight dict with keys "news", "macro", "polymarket".
                 Must sum to ≤ 1.0.  Defaults to {"news":0.4,"macro":0.3,"poly":0.3}.

    Returns:
        Q: np.ndarray of shape (N,) — weekly expected excess returns per sector.
        Omega: np.ndarray of shape (N, N) — diagonal view-uncertainty matrix.

    Raises:
        ValueError: No signal rows exist in the DB for this date.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS.copy()

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
    regime_scale = _RM_INTERCEPT + _RM_SLOPE * macro_regime  # ∈ [0.50, 1.00]

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

        # Conviction-weighted directional signal (macro regime doesn't supply
        # per-sector direction — only scales the magnitude).
        raw_signal = (
            weights["news"] * news_signal * news_conv
            + weights["polymarket"] * poly_signal * poly_conf
        )
        scaled_signal = raw_signal * regime_scale

        # Convert [-1, 1] → weekly excess return.
        q = scaled_signal * _MAX_EXCESS_RETURN_WEEKLY
        q_values.append(q)

        # Aggregate conviction across all three agents, dampened by macro regime.
        agg_conviction = (
            weights["news"] * news_conv
            + weights["macro"] * macro_conf
            + weights["polymarket"] * poly_conf
        ) * regime_scale

        # View uncertainty: high conviction → small omega → view carries more weight.
        omega_entry = _OMEGA_BASE / max(agg_conviction, _MIN_CONVICTION)
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
        "build_views date=%s  |Q|_max=%.5f  regime=%.1f  scale=%.2f  sectors=%d",
        date,
        float(np.abs(q_arr).max()) if q_arr.size else 0.0,
        macro_regime,
        regime_scale,
        n,
    )

    return q_arr, omega_arr
