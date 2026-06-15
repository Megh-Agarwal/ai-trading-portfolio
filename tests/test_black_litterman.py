"""Tests for src/optimizer/black_litterman.py — Ticket 3.2."""
from __future__ import annotations

import numpy as np
import pytest

from optimizer.black_litterman import black_litterman_posterior, build_picking_matrix

_N = 10


# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    n: int = _N,
    seed: int = 42,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (sigma, pi, Q, P, omega, tau) with realistic magnitudes."""
    rng = np.random.default_rng(seed)
    # PSD covariance via Wishart-like construction (always positive definite)
    A = rng.standard_normal((n + 5, n))
    sigma = A.T @ A / (n + 5) * 0.04   # ~4% annual variance on diagonal
    sigma = (sigma + sigma.T) / 2       # enforce exact symmetry

    pi = rng.uniform(0.03, 0.12, size=n)
    Q = pi + rng.uniform(-0.02, 0.02, size=n)
    P = build_picking_matrix(n)
    omega = np.diag(rng.uniform(1e-4, 1e-3, size=n))
    return sigma, pi, Q, P, omega, tau


# ---------------------------------------------------------------------------
# build_picking_matrix
# ---------------------------------------------------------------------------


class TestBuildPickingMatrix:
    def test_is_identity(self) -> None:
        P = build_picking_matrix(_N)
        np.testing.assert_array_equal(P, np.eye(_N))

    def test_shape(self) -> None:
        assert build_picking_matrix(_N).shape == (_N, _N)

    def test_float_dtype(self) -> None:
        assert np.issubdtype(build_picking_matrix(4).dtype, np.floating)

    def test_scalar_n_1(self) -> None:
        np.testing.assert_array_equal(build_picking_matrix(1), np.array([[1.0]]))


# ---------------------------------------------------------------------------
# Output shapes and types
# ---------------------------------------------------------------------------


class TestBLPosteriorShapes:
    def test_mu_posterior_shape(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        mu, _ = black_litterman_posterior(pi, sigma, P, Q, omega, tau)
        assert mu.shape == (_N,)

    def test_sigma_posterior_shape(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        _, s = black_litterman_posterior(pi, sigma, P, Q, omega, tau)
        assert s.shape == (_N, _N)

    def test_sigma_posterior_symmetric(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        _, s = black_litterman_posterior(pi, sigma, P, Q, omega, tau)
        np.testing.assert_allclose(s, s.T, atol=1e-12)

    def test_sigma_posterior_psd(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        _, s = black_litterman_posterior(pi, sigma, P, Q, omega, tau)
        eigvals = np.linalg.eigvalsh(s)
        assert np.all(eigvals >= -1e-8), f"sigma_posterior not PSD; min={eigvals.min():.2e}"

    def test_returns_numpy_arrays(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        mu, s = black_litterman_posterior(pi, sigma, P, Q, omega, tau)
        assert isinstance(mu, np.ndarray)
        assert isinstance(s, np.ndarray)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestBLPosteriorValidation:
    def test_raises_on_nonpositive_tau(self) -> None:
        sigma, pi, Q, P, omega, _ = _make_inputs()
        with pytest.raises(ValueError, match="tau must be positive"):
            black_litterman_posterior(pi, sigma, P, Q, omega, tau=0.0)

    def test_raises_on_sigma_shape_mismatch(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        with pytest.raises(ValueError, match="sigma must be"):
            black_litterman_posterior(pi, sigma[:5, :5], P, Q, omega, tau)

    def test_raises_on_omega_shape_mismatch(self) -> None:
        sigma, pi, Q, P, omega, tau = _make_inputs()
        with pytest.raises(ValueError, match="omega must be"):
            black_litterman_posterior(pi, sigma, P, Q, omega[:3, :3], tau)


# ---------------------------------------------------------------------------
# Mandatory sanity checks (ticket requirement)
# ---------------------------------------------------------------------------


class TestBLSanityChecks:
    def test_zero_views_posterior_equals_prior(self) -> None:
        """When Q = π, the views confirm the prior, so μ* = π exactly.

        Algebraic proof: rhs = ((τΣ)⁻¹ + P'Ω⁻¹P) π = M π
        → μ* = M⁻¹ M π = π  (independent of τ and Ω).
        """
        sigma, pi, _, P, omega, tau = _make_inputs()
        mu, _ = black_litterman_posterior(pi, sigma, P, pi.copy(), omega, tau)
        np.testing.assert_allclose(
            mu, pi, atol=0.005,
            err_msg="Q=π must give μ*=π regardless of omega or tau",
        )

    def test_high_confidence_views_dominate(self) -> None:
        """When Ω → 0 (tiny uncertainty), μ* → Q (views dominate).

        As Ω → 0: P'Ω⁻¹P → ∞, M⁻¹ → 0, M⁻¹ P'Ω⁻¹ → P†
        → μ* → P† Q = Q  (for P = I).
        """
        sigma, pi, Q, P, _, tau = _make_inputs()
        omega_tiny = np.diag(np.full(_N, 1e-8))
        mu, _ = black_litterman_posterior(pi, sigma, P, Q, omega_tiny, tau)
        np.testing.assert_allclose(
            mu, Q, atol=0.01,
            err_msg="Near-zero Ω must push μ* toward the views Q",
        )

    def test_low_confidence_equilibrium_dominates(self) -> None:
        """When Ω → ∞ (huge uncertainty), μ* → π (equilibrium dominates).

        As Ω → ∞: P'Ω⁻¹P → 0, M → (τΣ)⁻¹, M⁻¹ → τΣ
        → μ* → τΣ (τΣ)⁻¹ π = π.
        """
        sigma, pi, Q, P, _, tau = _make_inputs()
        omega_huge = np.diag(np.full(_N, 1e8))
        mu, _ = black_litterman_posterior(pi, sigma, P, Q, omega_huge, tau)
        np.testing.assert_allclose(
            mu, pi, atol=0.005,
            err_msg="Huge Ω must leave μ* at equilibrium π",
        )


# ---------------------------------------------------------------------------
# Cross-validation against pyportfolioopt
# ---------------------------------------------------------------------------


class TestCrossValidationVsPyPortfolioOpt:
    def test_mu_posterior_matches_pyportfolioopt(self) -> None:
        from pypfopt.black_litterman import BlackLittermanModel

        sigma, pi, Q, P, omega, tau = _make_inputs(seed=7)
        mu_ours, _ = black_litterman_posterior(pi, sigma, P, Q, omega, tau)

        bl = BlackLittermanModel(sigma, pi=pi, Q=Q, P=P, omega=omega, tau=tau)
        mu_pypo = np.array(bl.bl_returns())

        np.testing.assert_allclose(
            mu_ours, mu_pypo, atol=1e-6,
            err_msg="mu_posterior must match pyportfolioopt to 1e-6",
        )

    def test_sigma_posterior_matches_pyportfolioopt(self) -> None:
        from pypfopt.black_litterman import BlackLittermanModel

        sigma, pi, Q, P, omega, tau = _make_inputs(seed=7)
        _, sigma_post_ours = black_litterman_posterior(pi, sigma, P, Q, omega, tau)

        bl = BlackLittermanModel(sigma, pi=pi, Q=Q, P=P, omega=omega, tau=tau)
        sigma_post_pypo = np.array(bl.bl_cov())

        np.testing.assert_allclose(
            sigma_post_ours, sigma_post_pypo, atol=1e-6,
            err_msg="sigma_posterior must match pyportfolioopt to 1e-6",
        )

    def test_cross_validation_multiple_seeds(self) -> None:
        """Spot-check 5 different random inputs to confirm robustness."""
        from pypfopt.black_litterman import BlackLittermanModel

        for seed in [1, 13, 42, 99, 137]:
            sigma, pi, Q, P, omega, tau = _make_inputs(seed=seed)
            mu_ours, _ = black_litterman_posterior(pi, sigma, P, Q, omega, tau)
            bl = BlackLittermanModel(sigma, pi=pi, Q=Q, P=P, omega=omega, tau=tau)
            mu_pypo = np.array(bl.bl_returns())
            np.testing.assert_allclose(
                mu_ours, mu_pypo, atol=1e-6,
                err_msg=f"mu_posterior mismatch at seed={seed}",
            )


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestBLConfig:
    def test_tau_in_config(self) -> None:
        from config import load_config
        cfg = load_config("optimizer")
        assert cfg.black_litterman.tau == pytest.approx(0.05)

    def test_tau_positive(self) -> None:
        from config import load_config
        cfg = load_config("optimizer")
        assert cfg.black_litterman.tau > 0
