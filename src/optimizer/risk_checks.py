"""Pre-trade and post-trade risk checks for portfolio rebalancing (ADR-018).

Checks run before and after optimization to catch constraint violations and
macro risk events. Each check returns a RiskCheckResult describing whether it
fired and what action was (or would be) taken. Orchestration and action
application happen in run_all_risk_checks.

Public API:
- check_max_position: verify no weight exceeds the position cap.
- check_max_turnover: verify single-rebalance turnover is within limits.
- check_drawdown_circuit_breaker: halt rebalancing if rolling drawdown is severe.
- check_realized_vol: partially deleverage if realized vol breaches threshold.
- run_all_risk_checks: run all checks, apply actions, log to DB.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from db.models import PortfolioSnapshot, RiskEvent
from sqlalchemy import select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

_SQRT_252 = np.sqrt(252.0)


@dataclass
class RiskCheckResult:
    check_name: str
    passed: bool
    value: float | None      # computed metric (e.g. drawdown, turnover)
    threshold: float | None  # limit that was compared against
    message: str
    action: str              # "none" if passed; action description if triggered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(d: str | datetime.date) -> datetime.date:
    return datetime.date.fromisoformat(d) if isinstance(d, str) else d


def _query_snapshot_values(
    rebalance_date: datetime.date,
    lookback: int,
    db: Engine,
) -> list[float]:
    """Return total_value rows up to rebalance_date, newest-first, capped at lookback."""
    with Session(db) as session:
        rows = session.execute(
            select(PortfolioSnapshot.total_value)
            .where(PortfolioSnapshot.date <= rebalance_date)
            .order_by(PortfolioSnapshot.date.desc())
            .limit(lookback)
        ).all()
    return [r[0] for r in reversed(rows)]  # chronological order


def _log_risk_events(
    date: datetime.date,
    results: list[RiskCheckResult],
    db: Engine,
) -> None:
    with Session(db) as session:
        for r in results:
            session.add(RiskEvent(
                date=date,
                check_name=r.check_name,
                triggered=not r.passed,
                value=r.value,
                threshold=r.threshold,
                action_taken=r.action,
                message=r.message,
            ))
        session.commit()


# ---------------------------------------------------------------------------
# Pre-trade structural checks (pure weight math)
# ---------------------------------------------------------------------------


def check_max_position(
    weights: np.ndarray,
    config,  # OptimizerConfig
) -> RiskCheckResult:
    """Verify no position weight exceeds portfolio.max_position_weight.

    Args:
        weights: Proposed weight vector, shape (n,).
        config: OptimizerConfig.

    Returns:
        RiskCheckResult with action="clip_and_renorm" if triggered.
    """
    max_weight = float(np.max(weights))
    threshold = config.portfolio.max_position_weight
    # 1e-6 tolerance absorbs floating-point artefacts from clip+renorm in optimizer
    passed = max_weight <= threshold + 1e-6

    if passed:
        return RiskCheckResult(
            check_name="max_position",
            passed=True,
            value=max_weight,
            threshold=threshold,
            message=f"Max position {max_weight:.4f} within limit {threshold:.4f}",
            action="none",
        )
    return RiskCheckResult(
        check_name="max_position",
        passed=False,
        value=max_weight,
        threshold=threshold,
        message=f"Max position {max_weight:.4f} exceeds limit {threshold:.4f}; will clip and renormalize",
        action="clip_and_renorm",
    )


def check_max_turnover(
    new_weights: np.ndarray,
    prev_weights: np.ndarray,
    config,  # OptimizerConfig
) -> RiskCheckResult:
    """Verify single-rebalance L1 turnover is within risk.max_single_rebalance_turnover.

    Args:
        new_weights: Proposed weight vector after optimization.
        prev_weights: Current weight vector before rebalancing.
        config: OptimizerConfig.

    Returns:
        RiskCheckResult with action="blend_50_50" if triggered.
    """
    turnover = float(np.sum(np.abs(new_weights - prev_weights)))
    threshold = config.risk.max_single_rebalance_turnover
    passed = turnover <= threshold

    if passed:
        return RiskCheckResult(
            check_name="max_turnover",
            passed=True,
            value=turnover,
            threshold=threshold,
            message=f"Turnover {turnover:.4f} within limit {threshold:.4f}",
            action="none",
        )
    return RiskCheckResult(
        check_name="max_turnover",
        passed=False,
        value=turnover,
        threshold=threshold,
        message=f"Turnover {turnover:.4f} exceeds limit {threshold:.4f}; will blend 50/50 with previous weights",
        action="blend_50_50",
    )


# ---------------------------------------------------------------------------
# Post-trade / state-level checks (require portfolio_snapshot history)
# ---------------------------------------------------------------------------


def check_drawdown_circuit_breaker(
    date: str | datetime.date,
    db: Engine,
    config,  # OptimizerConfig
) -> RiskCheckResult:
    """Halt rebalancing if rolling drawdown exceeds risk.max_drawdown_threshold.

    Computes peak-to-trough drawdown over the last drawdown_lookback_days
    portfolio snapshots. Passes trivially when there is insufficient history
    (fewer than 2 snapshots).

    Args:
        date: Rebalancing date — query snapshots up to this date.
        db: SQLAlchemy Engine.
        config: OptimizerConfig.

    Returns:
        RiskCheckResult with action="halt_rebalance" if triggered.
    """
    rebalance_date = _to_date(date)
    lookback = config.risk.drawdown_lookback_days
    max_dd = config.risk.max_drawdown_threshold
    threshold = -max_dd  # drawdown is negative; threshold is e.g. -0.15

    values = _query_snapshot_values(rebalance_date, lookback, db)

    if len(values) < 2:
        return RiskCheckResult(
            check_name="drawdown_circuit_breaker",
            passed=True,
            value=None,
            threshold=threshold,
            message="Insufficient portfolio history — skipping drawdown check",
            action="none",
        )

    peak = max(values)
    current = values[-1]
    drawdown = (current - peak) / peak

    if drawdown < threshold:
        logger.warning(
            "Drawdown circuit breaker: rolling drawdown %.2f%% exceeds threshold %.2f%%",
            drawdown * 100,
            threshold * 100,
        )
        return RiskCheckResult(
            check_name="drawdown_circuit_breaker",
            passed=False,
            value=drawdown,
            threshold=threshold,
            message=(
                f"Rolling {lookback}-day drawdown {drawdown:.2%} exceeds "
                f"threshold {threshold:.2%}; halting rebalance"
            ),
            action="halt_rebalance",
        )

    return RiskCheckResult(
        check_name="drawdown_circuit_breaker",
        passed=True,
        value=drawdown,
        threshold=threshold,
        message=f"Rolling {lookback}-day drawdown {drawdown:.2%} within threshold {threshold:.2%}",
        action="none",
    )


def check_realized_vol(
    date: str | datetime.date,
    db: Engine,
    config,  # OptimizerConfig
) -> RiskCheckResult:
    """Trigger partial deleveraging if realized portfolio vol exceeds threshold.

    Computes annualised realised vol from daily returns over the last
    drawdown_lookback_days portfolio snapshots. Passes trivially when fewer
    than 3 snapshots exist (need ≥ 2 returns for a meaningful std estimate).

    Args:
        date: Rebalancing date — query snapshots up to this date.
        db: SQLAlchemy Engine.
        config: OptimizerConfig.

    Returns:
        RiskCheckResult with action="deleverage_20pct" if triggered.
    """
    rebalance_date = _to_date(date)
    lookback = config.risk.drawdown_lookback_days
    vol_threshold = config.portfolio.vol_target * config.risk.vol_breach_multiplier

    values = _query_snapshot_values(rebalance_date, lookback, db)

    if len(values) < 3:  # need ≥ 2 returns for std(ddof=1)
        return RiskCheckResult(
            check_name="realized_vol",
            passed=True,
            value=None,
            threshold=vol_threshold,
            message="Insufficient portfolio history — skipping realized vol check",
            action="none",
        )

    arr = np.array(values)
    returns = np.diff(arr) / arr[:-1]
    realized_vol = float(np.std(returns, ddof=1)) * _SQRT_252

    if realized_vol > vol_threshold:
        logger.warning(
            "Realized vol %.2f%% exceeds threshold %.2f%% — partial deleveraging",
            realized_vol * 100,
            vol_threshold * 100,
        )
        return RiskCheckResult(
            check_name="realized_vol",
            passed=False,
            value=realized_vol,
            threshold=vol_threshold,
            message=(
                f"Realized vol {realized_vol:.2%} exceeds threshold {vol_threshold:.2%} "
                f"({config.risk.vol_breach_multiplier}× vol_target); will deleverage"
            ),
            action="deleverage_20pct",
        )

    return RiskCheckResult(
        check_name="realized_vol",
        passed=True,
        value=realized_vol,
        threshold=vol_threshold,
        message=f"Realized vol {realized_vol:.2%} within threshold {vol_threshold:.2%}",
        action="none",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_risk_checks(
    date: str | datetime.date,
    new_weights: np.ndarray,
    prev_weights: np.ndarray,
    db: Engine,
    config,  # OptimizerConfig
) -> tuple[np.ndarray, list[RiskCheckResult]]:
    """Run all risk checks, apply triggered actions, and log results to DB.

    Execution order:
    1. check_drawdown_circuit_breaker — circuit breaker; if triggered, skip
       all other actions and return prev_weights unchanged.
    2. check_realized_vol — blend toward equal-weight if vol is too high.
    3. check_max_position — clip and renorm if any weight exceeds the cap.
    4. check_max_turnover — blend 50/50 with prev_weights if turnover is too high.

    All four results are logged to risk_events regardless of which fired.

    Args:
        date: Rebalancing date.
        new_weights: Proposed weight vector from optimizer.
        prev_weights: Weight vector before this rebalance.
        db: SQLAlchemy Engine.
        config: OptimizerConfig.

    Returns:
        (final_weights, results) where final_weights reflects all applied actions.
    """
    rebalance_date = _to_date(date)

    # Run all checks against original new_weights (collect before mutating)
    drawdown_result = check_drawdown_circuit_breaker(rebalance_date, db, config)
    vol_result = check_realized_vol(rebalance_date, db, config)
    pos_result = check_max_position(new_weights, config)
    turnover_result = check_max_turnover(new_weights, prev_weights, config)
    results = [drawdown_result, vol_result, pos_result, turnover_result]

    # Log everything before applying actions
    _log_risk_events(rebalance_date, results, db)

    # Circuit breaker: halt rebalance entirely
    if not drawdown_result.passed:
        return prev_weights.copy(), results

    working = new_weights.copy()

    # Vol breach: blend toward equal weight (partial deleveraging)
    if not vol_result.passed:
        n = len(working)
        equal_w = np.ones(n) / n
        blend = config.risk.vol_deleveraging_blend
        working = (1.0 - blend) * working + blend * equal_w
        working /= working.sum()

    # Position cap: clip and renorm
    if not pos_result.passed:
        max_pos = config.portfolio.max_position_weight
        working = np.clip(working, 0.0, max_pos)
        working /= working.sum()

    # Turnover cap: blend with previous weights
    if not turnover_result.passed:
        working = 0.5 * working + 0.5 * prev_weights
        working /= working.sum()

    return working, results
