"""Constrained mean-variance portfolio optimizer using CVXPY.

Optimization problem (maximise):
    μ @ w  -  (λ/2) × w @ Σ @ w  -  γ × ‖w − w_prev‖₁

Subject to:
    Σ w  = 1
    w_i ≥ 0
    w_i ≤ max_position_weight
    w @ Σ @ w ≤ vol_target²

Public API:
- optimize_weights: solve for optimal weights given BL posterior.
- compute_turnover: L1 norm of weight changes.
- compute_expected_portfolio_metrics: expected return, vol, Sharpe.
"""
from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def optimize_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    prev_weights: np.ndarray,
    config,  # OptimizerConfig — typed at runtime but kept loose to avoid top-level import
) -> np.ndarray:
    """Solve the constrained mean-variance problem and return optimal weights.

    Tries config.portfolio.solver_primary first; falls back to
    config.portfolio.solver_fallback on any failure. If both solvers fail,
    logs a WARNING and returns prev_weights unchanged (safe fallback — the
    portfolio holds its current allocation rather than making a bad trade).

    Args:
        mu: Posterior expected return vector, shape (n,). Annualised.
        sigma: Posterior covariance matrix, shape (n, n). Must be PSD.
        prev_weights: Current portfolio weights, shape (n,). Sums to 1.
        config: OptimizerConfig loaded from optimizer.yaml.

    Returns:
        Optimal weight vector, shape (n,). Guaranteed to:
        - sum to 1.0 ± 1e-6
        - lie in [0, max_position_weight] element-wise
    """
    n = len(mu)
    risk_aversion: float = config.risk_aversion
    pf = config.portfolio
    max_pos: float = pf.max_position_weight
    vol_target: float = pf.vol_target
    turnover_penalty: float = pf.turnover_penalty
    solver_primary: str = pf.solver_primary
    solver_fallback: str = pf.solver_fallback

    # Enforce exact symmetry — minor floating-point asymmetry can break quad_form
    sigma_sym = (sigma + sigma.T) / 2.0

    w = cp.Variable(n, name="weights")
    port_var = cp.quad_form(w, sigma_sym)

    objective = cp.Maximize(
        mu @ w
        - (risk_aversion / 2.0) * port_var
        - turnover_penalty * cp.norm1(w - prev_weights)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_pos,
        port_var <= vol_target ** 2,
    ]
    problem = cp.Problem(objective, constraints)

    def _try_solve(solver: str) -> bool:
        try:
            problem.solve(solver=solver, verbose=False)
            return problem.status in {"optimal", "optimal_inaccurate"}
        except Exception as exc:
            logger.debug("Solver %s raised %s: %s", solver, type(exc).__name__, exc)
            return False

    if not _try_solve(solver_primary):
        logger.warning(
            "Primary solver %s failed (status=%s); trying %s",
            solver_primary, problem.status, solver_fallback,
        )
        if not _try_solve(solver_fallback):
            logger.warning(
                "Fallback solver %s also failed (status=%s); returning prev_weights unchanged",
                solver_fallback, problem.status,
            )
            return prev_weights.copy()

    if w.value is None:
        logger.warning("Solver returned None weights; returning prev_weights unchanged")
        return prev_weights.copy()

    # Clip and renormalize to absorb minor solver inaccuracies
    weights = np.asarray(w.value, dtype=float).clip(0.0, max_pos)
    weights /= weights.sum()

    assert abs(weights.sum() - 1.0) < 1e-6, f"weights sum {weights.sum():.8f} ≠ 1"
    assert np.all(weights >= -1e-8), f"negative weight: {weights.min():.4e}"
    assert np.all(weights <= max_pos + 1e-6), f"weight exceeds cap {max_pos}: {weights.max():.4f}"

    logger.info(
        "optimize_weights  n=%d  turnover=%.4f  E[r]=%.3f%%  E[σ]=%.3f%%",
        n,
        float(np.sum(np.abs(weights - prev_weights))),
        float(mu @ weights * 100),
        float(np.sqrt(weights @ sigma_sym @ weights) * 100),
    )

    return weights


def compute_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """L1 norm of weight changes.

    Returns:
        float in [0.0, 2.0]. 0 = no trades; 2 = complete portfolio flip.
    """
    return float(np.sum(np.abs(new_weights - old_weights)))


def compute_expected_portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict:
    """Annualised expected return, volatility, and Sharpe (risk-free rate = 0).

    Args:
        weights: Portfolio weight vector, shape (n,).
        mu: Annualised expected return vector, shape (n,).
        sigma: Annualised covariance matrix, shape (n, n).

    Returns:
        Dict with keys expected_return, expected_vol, expected_sharpe (all float).
    """
    expected_return = float(mu @ weights)
    expected_vol = float(np.sqrt(weights @ sigma @ weights))
    expected_sharpe = expected_return / expected_vol if expected_vol > 1e-12 else 0.0
    return {
        "expected_return": expected_return,
        "expected_vol": expected_vol,
        "expected_sharpe": expected_sharpe,
    }
