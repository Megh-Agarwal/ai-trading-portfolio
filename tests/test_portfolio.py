"""Tests for src/optimizer/portfolio.py — Ticket 3.3."""
from __future__ import annotations

import numpy as np
import pytest

from optimizer.portfolio import (
    compute_expected_portfolio_metrics,
    compute_turnover,
    optimize_weights,
)

_N = 10


# ---------------------------------------------------------------------------
# Test config helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    vol_target: float = 0.95,
    turnover_penalty: float = 0.10,
    max_position_weight: float = 0.25,
    risk_aversion: float = 2.5,
    solver_primary: str = "CLARABEL",
    solver_fallback: str = "SCS",
):
    """Return a minimal OptimizerConfig for portfolio tests."""
    from config import load_config, PortfolioConfig
    cfg = load_config("optimizer")
    return cfg.model_copy(update={
        "risk_aversion": risk_aversion,
        "portfolio": PortfolioConfig(
            max_position_weight=max_position_weight,
            vol_target=vol_target,
            turnover_penalty=turnover_penalty,
            solver_primary=solver_primary,
            solver_fallback=solver_fallback,
        ),
    })


def _make_sigma(n: int = _N, seed: int = 42, scale: float = 0.04) -> np.ndarray:
    """PSD covariance matrix with realistic scale (~20% annual vol per asset)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n + 5, n))
    s = A.T @ A / (n + 5) * scale
    return (s + s.T) / 2


# ---------------------------------------------------------------------------
# Output shape / invariants
# ---------------------------------------------------------------------------


class TestOptimizeWeightsInvariants:
    def test_weights_sum_to_one(self) -> None:
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        w = optimize_weights(mu, sigma, prev, _make_config())
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_shape(self) -> None:
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        w = optimize_weights(mu, sigma, prev, _make_config())
        assert w.shape == (_N,)

    def test_weights_non_negative(self) -> None:
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        w = optimize_weights(mu, sigma, prev, _make_config())
        assert np.all(w >= -1e-8)

    def test_weights_within_cap(self) -> None:
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        cfg = _make_config(max_position_weight=0.25)
        w = optimize_weights(mu, sigma, prev, cfg)
        assert np.all(w <= 0.25 + 1e-6)

    def test_returns_numpy_array(self) -> None:
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        w = optimize_weights(mu, sigma, prev, _make_config())
        assert isinstance(w, np.ndarray)


# ---------------------------------------------------------------------------
# Behavioral tests (all four ticket requirements)
# ---------------------------------------------------------------------------


class TestBehavioralChecks:
    def test_neutral_views_near_market_weights(self) -> None:
        """Equilibrium mu → optimizer stays at market weights (within 5% per sector).

        When mu = λΣw_market, the unconstrained optimal is w_market by definition.
        With vol constraint loosened and no turnover incentive to move, the
        constrained solution stays at w_market.
        """
        from config import load_config

        # Market weights that respect max_position_weight — no sector above 25%
        market_weights = np.array([
            0.20, 0.15, 0.12, 0.10, 0.10, 0.09, 0.08, 0.07, 0.05, 0.04,
        ])
        assert abs(market_weights.sum() - 1.0) < 1e-9

        sigma = _make_sigma()
        risk_aversion = load_config("optimizer").risk_aversion
        # Equilibrium mu: the returns that make market_weights the MV-optimal portfolio
        pi = risk_aversion * sigma @ market_weights

        # Use a loose vol target so the constraint doesn't interfere
        actual_vol = float(np.sqrt(market_weights @ sigma @ market_weights))
        cfg = _make_config(vol_target=min(0.95, actual_vol + 0.15), turnover_penalty=0.0)

        weights = optimize_weights(pi, sigma, market_weights, cfg)

        max_dev = float(np.max(np.abs(weights - market_weights)))
        assert max_dev < 0.05, (
            f"Max sector deviation {max_dev:.3f} > 5% with equilibrium mu; "
            f"optimizer should stay near market weights"
        )

    def test_strong_bullish_xlk_hits_cap(self) -> None:
        """With XLK expected return at 15%, its weight should approach the 25% cap."""
        sigma = _make_sigma()
        # Equilibrium baseline, then spike asset 0 (represents XLK)
        from config import load_config
        market_weights = np.ones(_N) / _N
        pi = load_config("optimizer").risk_aversion * sigma @ market_weights
        mu = pi.copy()
        mu[0] = 0.15   # 15% expected return — well above equilibrium ~8%

        prev = np.ones(_N) / _N
        cfg = _make_config(vol_target=0.95, turnover_penalty=0.0)
        weights = optimize_weights(mu, sigma, prev, cfg)

        assert weights[0] >= 0.20, (
            f"Strong bullish signal (mu[0]=15%) should push asset 0 weight near cap; "
            f"got {weights[0]:.3f}"
        )

    def test_turnover_penalty_reduces_turnover(self) -> None:
        """Higher turnover penalty must produce meaningfully less turnover.

        Ticket spec: turnover(γ=0.5) < turnover(γ=0.0) × 0.7
        """
        sigma = _make_sigma(seed=7)
        # Strong signal away from equal weights
        mu = np.array([0.18, 0.16, 0.14, 0.12, 0.10, 0.04, 0.04, 0.04, 0.04, 0.04])
        prev = np.ones(_N) / _N  # equal weights — far from the alpha-optimal allocation

        cfg_no_pen = _make_config(vol_target=0.95, turnover_penalty=0.0)
        cfg_high_pen = _make_config(vol_target=0.95, turnover_penalty=0.5)

        w_no_pen = optimize_weights(mu, sigma, prev, cfg_no_pen)
        w_high_pen = optimize_weights(mu, sigma, prev, cfg_high_pen)

        to_no_pen = compute_turnover(prev, w_no_pen)
        to_high_pen = compute_turnover(prev, w_high_pen)

        assert to_no_pen > 0.05, f"Unconstrained turnover {to_no_pen:.4f} suspiciously low"
        assert to_high_pen < to_no_pen * 0.7, (
            f"High penalty turnover {to_high_pen:.4f} should be < 70% of "
            f"zero-penalty turnover {to_no_pen:.4f}"
        )

    def test_vol_target_binding(self) -> None:
        """When unconstrained solution exceeds vol_target, the constrained solution
        should satisfy w @ Σ @ w ≤ vol_target².

        Setup: high-vol assets (50% vol) with strong returns.  Unconstrained optimizer
        concentrates in them, pushing portfolio vol above 20%.  The vol constraint
        forces diversification into low-vol assets (10% vol).
        """
        n = 10
        # Assets 0-2: 50% annual vol; assets 3-9: 10% annual vol; no cross-correlation
        vols = np.array([0.50] * 3 + [0.10] * 7)
        sigma = np.diag(vols ** 2)

        # Only high-vol assets have alpha
        mu = np.array([0.20, 0.18, 0.16] + [0.03] * 7)
        prev = np.ones(n) / n
        vol_target = 0.20

        # Without vol constraint (loose target): optimizer concentrates in high-vol
        cfg_loose = _make_config(
            vol_target=0.95, turnover_penalty=0.0, max_position_weight=0.25,
        )
        w_loose = optimize_weights(mu, sigma, prev, cfg_loose)
        vol_loose = float(np.sqrt(w_loose @ sigma @ w_loose))

        assert vol_loose > vol_target, (
            f"Unconstrained vol {vol_loose:.3f} should exceed target {vol_target}; "
            "test setup is wrong if this fails"
        )

        # With vol constraint: must stay ≤ vol_target
        cfg_tight = _make_config(
            vol_target=vol_target, turnover_penalty=0.0, max_position_weight=0.25,
        )
        w_tight = optimize_weights(mu, sigma, prev, cfg_tight)
        vol_tight = float(np.sqrt(w_tight @ sigma @ w_tight))

        assert vol_tight <= vol_target + 0.001, (
            f"Constrained vol {vol_tight:.4f} must be ≤ vol_target+0.001={vol_target+0.001}"
        )


# ---------------------------------------------------------------------------
# Solver fallback
# ---------------------------------------------------------------------------


class TestSolverFallback:
    def test_fallback_to_scs_when_primary_fails(self) -> None:
        """When solver_primary is invalid, solver_fallback (SCS) should succeed."""
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        cfg = _make_config(solver_primary="FAKE_SOLVER", solver_fallback="SCS")

        weights = optimize_weights(mu, sigma, prev, cfg)

        assert abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights >= -1e-8)
        assert np.all(weights <= 0.25 + 1e-6)

    def test_returns_prev_weights_when_all_solvers_fail(self) -> None:
        """When both solvers fail, prev_weights is returned unchanged."""
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        prev = np.ones(_N) / _N
        cfg = _make_config(solver_primary="FAKE_A", solver_fallback="FAKE_B")

        weights = optimize_weights(mu, sigma, prev, cfg)

        np.testing.assert_array_equal(weights, prev)


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------


class TestComputeTurnover:
    def test_zero_turnover_identical_weights(self) -> None:
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert compute_turnover(w, w) == pytest.approx(0.0)

    def test_full_flip_gives_two(self) -> None:
        old = np.array([1.0, 0.0])
        new = np.array([0.0, 1.0])
        assert compute_turnover(old, new) == pytest.approx(2.0)

    def test_partial_rebalance(self) -> None:
        old = np.array([0.5, 0.5])
        new = np.array([0.6, 0.4])
        assert compute_turnover(old, new) == pytest.approx(0.2)

    def test_range_zero_to_two(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            w1 = rng.dirichlet(np.ones(10))
            w2 = rng.dirichlet(np.ones(10))
            to = compute_turnover(w1, w2)
            assert 0.0 <= to <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# compute_expected_portfolio_metrics
# ---------------------------------------------------------------------------


class TestComputeExpectedMetrics:
    def test_return_equals_weighted_average(self) -> None:
        weights = np.array([0.6, 0.4])
        mu = np.array([0.10, 0.05])
        sigma = np.eye(2) * 0.04
        m = compute_expected_portfolio_metrics(weights, mu, sigma)
        assert m["expected_return"] == pytest.approx(0.08, rel=1e-6)

    def test_vol_matches_formula(self) -> None:
        weights = np.array([0.5, 0.5])
        sigma = np.array([[0.04, 0.0], [0.0, 0.04]])
        mu = np.array([0.08, 0.08])
        m = compute_expected_portfolio_metrics(weights, mu, sigma)
        expected_vol = float(np.sqrt(weights @ sigma @ weights))
        assert m["expected_vol"] == pytest.approx(expected_vol, rel=1e-6)

    def test_sharpe_ratio(self) -> None:
        weights = np.array([1.0])
        mu = np.array([0.10])
        sigma = np.array([[0.04]])   # vol = 0.20
        m = compute_expected_portfolio_metrics(weights, mu, sigma)
        assert m["expected_sharpe"] == pytest.approx(0.10 / 0.20, rel=1e-6)

    def test_keys_present(self) -> None:
        weights = np.ones(_N) / _N
        sigma = _make_sigma()
        mu = np.ones(_N) * 0.08
        m = compute_expected_portfolio_metrics(weights, mu, sigma)
        assert {"expected_return", "expected_vol", "expected_sharpe"} == set(m)

    def test_zero_vol_sharpe_is_zero(self) -> None:
        """Degenerate case: zero-vol portfolio gives Sharpe = 0, not division error."""
        weights = np.array([1.0])
        mu = np.array([0.10])
        sigma = np.array([[0.0]])
        m = compute_expected_portfolio_metrics(weights, mu, sigma)
        assert m["expected_sharpe"] == 0.0


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestPortfolioConfig:
    def test_config_loads_portfolio_stanza(self) -> None:
        from config import load_config
        cfg = load_config("optimizer")
        assert cfg.portfolio.max_position_weight == pytest.approx(0.25)
        assert cfg.portfolio.vol_target == pytest.approx(0.12)
        assert cfg.portfolio.turnover_penalty == pytest.approx(0.10)
        assert cfg.portfolio.solver_primary == "CLARABEL"
        assert cfg.portfolio.solver_fallback == "SCS"
