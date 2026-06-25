"""Optimizer pipeline: sequences all M3 components into a single callable (Ticket 3.6).

End-to-end flow:
  load views → compute prior → BL posterior → constrained optimization →
  risk checks → write target weights → return summary dict

This is the function M4's weekly run script will call.
"""

from __future__ import annotations

import datetime
import logging
import traceback
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from config import load_config
from db.models import PORTFOLIO_LIVE, PortfolioSnapshot, RiskEvent, TargetWeight, View
from execution.costs import estimate_portfolio_rebalance_cost
from optimizer.black_litterman import black_litterman_posterior, build_picking_matrix
from optimizer.equilibrium import get_prior
from optimizer.portfolio import (
    compute_expected_portfolio_metrics,
    compute_turnover,
    optimize_weights,
)
from optimizer.risk_checks import run_all_risk_checks

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# Matches views.py — floor used when no conviction is stored
_MIN_CONVICTION: float = 0.01
# Matches views.py — used to convert weekly omega_base to annual variance
_WEEKS_PER_YEAR: int = 52
# Used when no portfolio snapshot exists yet (first run)
_FALLBACK_PORTFOLIO_VALUE: float = 1_000_000.0


def run_optimization_pipeline(
    date: str | datetime.date,
    db: Engine,
    mode: str = "backtest",
    portfolio_id: str = PORTFOLIO_LIVE,
) -> dict:
    """Run the full optimization pipeline for one rebalance date.

    Sequences: load views → compute prior (pi, Sigma) → BL posterior →
    constrained optimization → risk checks → write target weights → metrics.

    Args:
        date: Rebalance date. Views for this date must already be in the DB
            (written by src/agents/pipeline.py). ISO string or date object.
        db: SQLAlchemy Engine connected to state.db.
        mode: "backtest" or "live". Passed through to the summary dict as
            metadata; does not affect the BL math (views are already in DB).
        portfolio_id: Portfolio namespace. All reads and writes are scoped to
            this ID so parallel backtests do not collide.

    Returns:
        Summary dict with keys:
            date, weights, expected_return_annual, expected_vol_annual,
            turnover, estimated_cost_usd, risk_checks, any_risk_triggered,
            mode, views_available.

    Raises:
        Any unhandled exception is logged at CRITICAL level with full traceback
        and then re-raised. Optimizer failures are never swallowed silently.
    """
    try:
        return _run(date, db, mode, portfolio_id)
    except Exception:
        logger.critical(
            "Optimizer pipeline FAILED  date=%s  mode=%s  portfolio=%s\n%s",
            date,
            mode,
            portfolio_id,
            traceback.format_exc(),
        )
        raise


# ---------------------------------------------------------------------------
# Private implementation
# ---------------------------------------------------------------------------


def _run(date: str | datetime.date, db: Engine, mode: str, portfolio_id: str) -> dict:
    rebalance_date = datetime.date.fromisoformat(date) if isinstance(date, str) else date

    cfg = load_config("optimizer")
    tickers: list[str] = load_config("universe").ticker_list
    n = len(tickers)

    # Step 1 — load views (Q, Omega) from DB; fallback to zero-view equilibrium
    views_available, Q, omega = _load_views(rebalance_date, tickers, cfg, db, portfolio_id)

    # Step 2 — load previous week's target weights; equal weights on first run
    prev_weights = _load_prev_weights(rebalance_date, tickers, n, db, portfolio_id)

    # Step 3 — BL prior: equilibrium returns and covariance from price history
    pi, sigma = get_prior(rebalance_date, db)

    # Edge case: rare non-PSD covariance (sparse price data) → regularize
    sigma = _ensure_psd(sigma)

    # Step 4 — BL posterior: blend prior with agent views
    P = build_picking_matrix(n)
    tau = cfg.black_litterman.tau
    mu_post, sigma_post = black_litterman_posterior(pi, sigma, P, Q, omega, tau)

    # Step 5 — constrained mean-variance optimization
    candidate_weights, vol_constraint_status = optimize_weights(
        mu_post, sigma_post, prev_weights, cfg
    )
    _log_vol_constraint_event(
        rebalance_date, vol_constraint_status, candidate_weights, sigma_post, cfg, db, portfolio_id
    )

    # Step 6 — risk checks; circuit breaker may revert to prev_weights
    final_weights, risk_results = run_all_risk_checks(
        rebalance_date, candidate_weights, prev_weights, db, cfg, portfolio_id=portfolio_id
    )

    # Step 7 — persist target weights (idempotent: delete-before-insert)
    _write_target_weights(rebalance_date, tickers, final_weights, db, portfolio_id)

    # Step 8 — portfolio metrics and cost estimate
    metrics = compute_expected_portfolio_metrics(final_weights, mu_post, sigma_post)
    turnover = compute_turnover(prev_weights, final_weights)
    portfolio_value = _get_portfolio_value(rebalance_date, db, portfolio_id)
    estimated_cost = estimate_portfolio_rebalance_cost(
        prev_weights, final_weights, portfolio_value, cfg.transaction_costs
    )
    any_triggered = any(not r.passed for r in risk_results)

    logger.info(
        "run_optimization_pipeline  date=%s  mode=%s  "
        "E[r]=%.2f%%  E[σ]=%.2f%%  turnover=%.4f  cost=$%.2f  "
        "risk_triggered=%s  views=%s  vol_status=%s",
        rebalance_date,
        mode,
        metrics["expected_return"] * 100,
        metrics["expected_vol"] * 100,
        turnover,
        estimated_cost,
        any_triggered,
        views_available,
        vol_constraint_status,
    )

    return {
        "date": str(rebalance_date),
        "weights": {t: float(w) for t, w in zip(tickers, final_weights)},
        "expected_return_annual": metrics["expected_return"],
        "expected_vol_annual": metrics["expected_vol"],
        "turnover": turnover,
        "estimated_cost_usd": estimated_cost,
        "risk_checks": risk_results,
        "any_risk_triggered": any_triggered,
        "mode": mode,
        "views_available": views_available,
        "vol_constraint_status": vol_constraint_status,
    }


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _load_views(
    date: datetime.date,
    tickers: list[str],
    cfg,
    db: Engine,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Load Q and Omega from the views table for portfolio_id.

    Omega is reconstructed from stored confidence using the same formula as
    views.py: omega_i = omega_base * _WEEKS_PER_YEAR / max(confidence_i, _MIN_CONVICTION).

    Returns (views_available, Q, Omega).
    """
    with Session(db) as session:
        rows = session.execute(
            select(View.sector, View.expected_return, View.confidence)
            .where(View.portfolio_id == portfolio_id)
            .where(View.date == date)
        ).all()

    omega_base: float = cfg.aggregator.omega_base
    large_uncertainty = omega_base * _WEEKS_PER_YEAR / _MIN_CONVICTION

    if not rows:
        logger.warning(
            "No views in DB for date=%s — using zero views; "
            "posterior will be close to equilibrium weights",
            date,
        )
        n = len(tickers)
        Q = np.zeros(n, dtype=float)
        omega = np.diag([large_uncertainty] * n)
        return False, Q, omega

    view_by_sector = {r[0]: (r[1], r[2]) for r in rows}

    q_values: list[float] = []
    omega_diag: list[float] = []
    for ticker in tickers:
        if ticker in view_by_sector:
            q, conf = view_by_sector[ticker]
            q_values.append(float(q) if q is not None else 0.0)
            conviction = float(conf) if conf is not None else 0.0
            omega_diag.append(omega_base * _WEEKS_PER_YEAR / max(conviction, _MIN_CONVICTION))
        else:
            # Sector missing from views — treat as no-view (high uncertainty, zero Q)
            q_values.append(0.0)
            omega_diag.append(large_uncertainty)

    return True, np.array(q_values, dtype=float), np.diag(omega_diag)


def _load_prev_weights(
    date: datetime.date,
    tickers: list[str],
    n: int,
    db: Engine,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> np.ndarray:
    """Return the most recent target weights strictly before `date` for portfolio_id.

    Falls back to equal weights and logs INFO on first run (no history).
    """
    with Session(db) as session:
        prev_date = session.execute(
            select(TargetWeight.date)
            .where(TargetWeight.portfolio_id == portfolio_id)
            .where(TargetWeight.date < date)
            .order_by(TargetWeight.date.desc())
            .limit(1)
        ).scalar()

        if prev_date is None:
            logger.info(
                "No previous target weights found for portfolio=%s — using equal weights (1/n) "
                "as starting point for date=%s",
                portfolio_id,
                date,
            )
            return np.ones(n, dtype=float) / n

        rows = session.execute(
            select(TargetWeight.sector, TargetWeight.weight)
            .where(TargetWeight.portfolio_id == portfolio_id)
            .where(TargetWeight.date == prev_date)
        ).all()

    weight_by_sector = {r[0]: float(r[1]) for r in rows}
    weights = np.array([weight_by_sector.get(t, 1.0 / n) for t in tickers], dtype=float)
    # Renormalize in case stored weights have minor floating-point drift
    weights /= weights.sum()
    return weights


def _ensure_psd(sigma: np.ndarray) -> np.ndarray:
    """Apply 1e-6 × I regularization if sigma is not positive semi-definite.

    This can occur with very sparse price histories (Ledoit-Wolf still returns
    a valid estimate but floating-point accumulation can break Cholesky).
    """
    try:
        np.linalg.cholesky(sigma)
        return sigma
    except np.linalg.LinAlgError:
        logger.warning(
            "Covariance matrix is not PSD (sparse price data?) — applying 1e-6 × I regularization"
        )
        return sigma + 1e-6 * np.eye(sigma.shape[0])


def _write_target_weights(
    date: datetime.date,
    tickers: list[str],
    weights: np.ndarray,
    db: Engine,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> None:
    """Upsert final weights into target_weights (delete-before-insert, scoped to portfolio_id)."""
    with Session(db) as session:
        session.execute(
            delete(TargetWeight)
            .where(TargetWeight.portfolio_id == portfolio_id)
            .where(TargetWeight.date == date)
        )
        for ticker, w in zip(tickers, weights):
            session.add(
                TargetWeight(portfolio_id=portfolio_id, date=date, sector=ticker, weight=float(w))
            )
        session.commit()


def _log_vol_constraint_event(
    date: datetime.date,
    status: str,
    weights: np.ndarray,
    sigma: np.ndarray,
    cfg,
    db: Engine,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> None:
    """Write vol_constraint status as a risk_events row for every rebalance."""
    actual_vol = float(np.sqrt(weights @ sigma @ weights))
    triggered = status == "infeasible_relaxed"
    with Session(db) as session:
        session.add(
            RiskEvent(
                portfolio_id=portfolio_id,
                date=date,
                check_name="vol_constraint",
                triggered=triggered,
                value=actual_vol,
                threshold=cfg.portfolio.vol_target,
                action_taken="relax_vol_constraint" if triggered else "none",
                message=(
                    f"vol_constraint_status={status}  "
                    f"actual_vol={actual_vol:.4f}  "
                    f"vol_target={cfg.portfolio.vol_target:.4f}"
                ),
            )
        )
        session.commit()


def _get_portfolio_value(
    date: datetime.date,
    db: Engine,
    portfolio_id: str = PORTFOLIO_LIVE,
) -> float:
    """Return the most recent portfolio total value up to `date` for portfolio_id.

    Falls back to _FALLBACK_PORTFOLIO_VALUE when no snapshot exists yet
    (first run before M4 execution layer writes snapshots).
    """
    with Session(db) as session:
        snap = session.execute(
            select(PortfolioSnapshot.total_value)
            .where(PortfolioSnapshot.portfolio_id == portfolio_id)
            .where(PortfolioSnapshot.date <= date)
            .order_by(PortfolioSnapshot.date.desc())
            .limit(1)
        ).scalar()
    return float(snap) if snap is not None else _FALLBACK_PORTFOLIO_VALUE
