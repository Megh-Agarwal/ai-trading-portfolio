"""Constrained mean-variance portfolio optimizer using CVXPY.

Optimization problem (maximise):
    μ @ w  -  (λ/2) × w @ Σ @ w  -  γ × ‖w − w_prev‖₁

Subject to:
    Σ w  = 1
    w_i ≥ 0
    w_i ≤ max_position_weight
    w @ Σ @ w ≤ vol_target²   (relaxed when infeasible — see vol_constraint_status)

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
) -> tuple[np.ndarray, str]:
    """Solve the constrained mean-variance problem and return optimal weights.

    Tries config.portfolio.solver_primary first; falls back to
    config.portfolio.solver_fallback. If the vol constraint makes the problem
    infeasible, the constraint is dropped and the problem is re-solved (this
    finds the best feasible portfolio under the remaining constraints). Only if
    all solvers fail on the relaxed problem are prev_weights returned.

    Args:
        mu: Posterior expected return vector, shape (n,). Annualised.
        sigma: Posterior covariance matrix, shape (n, n). Must be PSD.
        prev_weights: Current portfolio weights, shape (n,). Sums to 1.
        config: OptimizerConfig loaded from optimizer.yaml.

    Returns:
        (weights, vol_constraint_status) where:
          weights: Optimal weight vector, shape (n,). Sums to 1 ± 1e-6,
                   lies in [0, max_position_weight] element-wise.
          vol_constraint_status: one of
            "not_binding"       — vol constraint had headroom; did not limit solution
            "binding"           — vol constraint was active and limited the solution
            "infeasible_relaxed"— vol_target unreachable; constraint dropped, re-solved
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

    def _build_problem(include_vol_constraint: bool) -> tuple[cp.Problem, cp.Variable]:
        wv = cp.Variable(n, name="weights")
        pv = cp.quad_form(wv, sigma_sym)
        obj = cp.Maximize(
            mu @ wv - (risk_aversion / 2.0) * pv - turnover_penalty * cp.norm1(wv - prev_weights)
        )
        constrs = [cp.sum(wv) == 1, wv >= 0, wv <= max_pos]
        if include_vol_constraint:
            constrs.append(pv <= vol_target**2)
        return cp.Problem(obj, constrs), wv

    def _try_solve(prob: cp.Problem, solver: str) -> bool:
        try:
            prob.solve(solver=solver, verbose=False)
            return prob.status in {"optimal", "optimal_inaccurate"}
        except Exception as exc:
            logger.debug("Solver %s raised %s: %s", solver, type(exc).__name__, exc)
            return False

    # --- First attempt: with vol constraint ---
    problem, w = _build_problem(include_vol_constraint=True)

    solved = _try_solve(problem, solver_primary)
    if not solved:
        logger.warning(
            "Primary solver %s failed (status=%s); trying %s",
            solver_primary,
            problem.status,
            solver_fallback,
        )
        solved = _try_solve(problem, solver_fallback)

    if not solved:
        # Vol target is unreachable (or numerical failure) — drop constraint and re-solve
        logger.warning(
            "Fallback solver %s also failed (status=%s); "
            "vol_target=%.2f%% unreachable — dropping vol constraint and re-solving",
            solver_fallback,
            problem.status,
            vol_target * 100,
        )
        problem2, w2 = _build_problem(include_vol_constraint=False)

        solved2 = _try_solve(problem2, solver_primary)
        if not solved2:
            logger.warning(
                "Primary solver %s failed on relaxed problem; trying %s",
                solver_primary,
                solver_fallback,
            )
            solved2 = _try_solve(problem2, solver_fallback)

        if not solved2 or w2.value is None:
            logger.warning(
                "All solvers failed even without vol constraint; returning prev_weights unchanged"
            )
            return prev_weights.copy(), "infeasible_relaxed"

        weights = np.asarray(w2.value, dtype=float).clip(0.0, max_pos)
        weights /= weights.sum()

        assert abs(weights.sum() - 1.0) < 1e-6, f"weights sum {weights.sum():.8f} ≠ 1"
        assert np.all(weights >= -1e-8), f"negative weight: {weights.min():.4e}"
        assert np.all(weights <= max_pos + 1e-6), (
            f"weight exceeds cap {max_pos}: {weights.max():.4f}"
        )

        logger.warning(
            "optimize_weights[infeasible_relaxed]  n=%d  vol_target=%.2f%%  "
            "actual_vol=%.3f%%  turnover=%.4f  E[r]=%.3f%%",
            n,
            vol_target * 100,
            float(np.sqrt(weights @ sigma_sym @ weights) * 100),
            float(np.sum(np.abs(weights - prev_weights))),
            float(mu @ weights * 100),
        )
        return weights, "infeasible_relaxed"

    if w.value is None:
        logger.warning("Solver returned None weights; returning prev_weights unchanged")
        return prev_weights.copy(), "not_binding"

    # Clip and renormalize to absorb minor solver inaccuracies
    weights = np.asarray(w.value, dtype=float).clip(0.0, max_pos)
    weights /= weights.sum()

    # Binding if the achieved vol is within 0.1% of the target — the optimizer
    # pressed against the constraint wall
    actual_vol = float(np.sqrt(weights @ sigma_sym @ weights))
    vol_status = "binding" if actual_vol >= vol_target - 0.001 else "not_binding"

    assert abs(weights.sum() - 1.0) < 1e-6, f"weights sum {weights.sum():.8f} ≠ 1"
    assert np.all(weights >= -1e-8), f"negative weight: {weights.min():.4e}"
    assert np.all(weights <= max_pos + 1e-6), f"weight exceeds cap {max_pos}: {weights.max():.4f}"

    logger.info(
        "optimize_weights[%s]  n=%d  turnover=%.4f  E[r]=%.3f%%  E[σ]=%.3f%%",
        vol_status,
        n,
        float(np.sum(np.abs(weights - prev_weights))),
        float(mu @ weights * 100),
        actual_vol * 100,
    )

    return weights, vol_status


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
