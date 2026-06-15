"""Black-Litterman posterior: blend equilibrium prior with agent-derived views.

Public API:
- build_picking_matrix: identity matrix P for one-view-per-sector case.
- black_litterman_posterior: compute (mu_posterior, sigma_posterior) from BL formulas.

Math reference (He & Litterman 1999):
    M  = (τΣ)⁻¹ + P'Ω⁻¹P
    μ* = M⁻¹ × ( (τΣ)⁻¹π + P'Ω⁻¹Q )
    Σ* = M⁻¹ + Σ

All linear systems are solved with np.linalg.solve rather than explicit matrix
inversion to improve numerical stability.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_picking_matrix(n_assets: int) -> np.ndarray:
    """Return the P (picking) matrix for one-view-per-sector case.

    Args:
        n_assets: Number of sectors / assets in the universe.

    Returns:
        Identity matrix of shape (n_assets, n_assets). Each row selects one
        sector's absolute return view, so P = I is the natural choice when every
        sector has exactly one view. Kept as a separate function for extensibility
        — relative views (row sums to zero) are straightforward in v2.
    """
    return np.eye(n_assets)


def black_litterman_posterior(
    pi: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend equilibrium prior with agent views to produce BL posterior.

    Implements the standard He-Litterman formulas:
        M  = (τΣ)⁻¹ + P'Ω⁻¹P
        μ* = M⁻¹ × ( (τΣ)⁻¹π + P'Ω⁻¹Q )
        Σ* = M⁻¹ + Σ

    All solves use np.linalg.solve; no np.linalg.inv calls.

    Args:
        pi: Equilibrium return vector, shape (n,).
        sigma: Annualised prior covariance matrix, shape (n, n). Must be PSD.
        P: Picking matrix, shape (k, n). k = number of views. Use
            build_picking_matrix(n) for the one-view-per-sector case.
        Q: View return vector, shape (k,). From aggregator.build_views().
        omega: View uncertainty matrix, shape (k, k). Diagonal. From
            aggregator.build_views().
        tau: Prior uncertainty scalar — how much τ scales Σ. Typically 0.05.
            Smaller τ means more confidence in the equilibrium prior.

    Returns:
        (mu_posterior, sigma_posterior):
            mu_posterior — posterior expected return vector, shape (n,).
            sigma_posterior — posterior covariance matrix, shape (n, n).

    Raises:
        ValueError: Shape mismatches or non-positive tau.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    n = pi.shape[0]
    k = Q.shape[0]

    if sigma.shape != (n, n):
        raise ValueError(f"sigma must be ({n}, {n}), got {sigma.shape}")
    if P.shape != (k, n):
        raise ValueError(f"P must be ({k}, {n}), got {P.shape}")
    if omega.shape != (k, k):
        raise ValueError(f"omega must be ({k}, {k}), got {omega.shape}")

    tau_sigma = tau * sigma

    # (τΣ)⁻¹ as a matrix — needed to form M and the RHS first term.
    # Solved as: tau_sigma @ X = I  →  X = (τΣ)⁻¹
    tau_sigma_inv = np.linalg.solve(tau_sigma, np.eye(n))

    # Ω⁻¹ — view precision matrix.
    # Solved as: omega @ X = I  →  X = Ω⁻¹
    omega_inv = np.linalg.solve(omega, np.eye(k))

    # M = (τΣ)⁻¹ + P'Ω⁻¹P
    M = tau_sigma_inv + P.T @ omega_inv @ P

    # RHS = (τΣ)⁻¹π + P'Ω⁻¹Q
    rhs = tau_sigma_inv @ pi + P.T @ (omega_inv @ Q)

    # μ* = M⁻¹ × rhs — solved directly without forming M⁻¹
    mu_posterior = np.linalg.solve(M, rhs)

    # Σ* = M⁻¹ + Σ — need M⁻¹ as a full matrix here
    M_inv = np.linalg.solve(M, np.eye(n))
    sigma_posterior = M_inv + sigma

    assert mu_posterior.shape == (n,), f"mu_posterior shape {mu_posterior.shape} != ({n},)"
    assert sigma_posterior.shape == (n, n), (
        f"sigma_posterior shape {sigma_posterior.shape} != ({n},{n})"
    )

    logger.info(
        "black_litterman_posterior  n=%d  k=%d  τ=%.3f  "
        "μ*=[%.1f%%, %.1f%%]  Σ*_diag=[%.3f%%, %.3f%%]",
        n, k, tau,
        float(mu_posterior.min() * 100), float(mu_posterior.max() * 100),
        float(np.sqrt(np.diag(sigma_posterior)).min() * 100),
        float(np.sqrt(np.diag(sigma_posterior)).max() * 100),
    )

    return mu_posterior, sigma_posterior
