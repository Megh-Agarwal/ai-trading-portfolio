"""Tests for src/optimizer/equilibrium.py — Ticket 3.1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.equilibrium import compute_covariance, compute_equilibrium_returns

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_N = 10
_N_DAYS = 350  # > 252 + 1, enough for full lookback


def _make_prices(
    n_tickers: int = _N,
    n_days: int = _N_DAYS,
    annual_vol: float = 0.20,
    annual_drift: float = 0.08,
    correlation: float = 0.70,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic price series with controlled vol and inter-asset correlation.

    All assets share a common factor (Cholesky decomposition of a constant
    correlation matrix) so that:
    - Each asset's annualised vol ≈ annual_vol
    - Pairwise correlation ≈ correlation
    - Equilibrium returns (with λ=2.5, equal weights) land in the 3-12% range
    """
    rng = np.random.default_rng(seed)
    corr_mat = correlation * np.ones((n_tickers, n_tickers)) + (1 - correlation) * np.eye(n_tickers)
    L = np.linalg.cholesky(corr_mat)

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252

    z = rng.standard_normal((n_days, n_tickers))
    log_ret = daily_drift + daily_vol * (z @ L.T)
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, columns=[f"ETF_{i}" for i in range(n_tickers)])


_EQUAL_WEIGHTS: dict[str, float] = {f"ETF_{i}": 1.0 for i in range(_N)}


# ---------------------------------------------------------------------------
# compute_covariance
# ---------------------------------------------------------------------------


class TestComputeCovariance:
    def test_shape(self) -> None:
        cov = compute_covariance(_make_prices(), lookback_days=252)
        assert cov.shape == (_N, _N)

    def test_symmetric(self) -> None:
        cov = compute_covariance(_make_prices(), lookback_days=252)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_positive_semidefinite(self) -> None:
        cov = compute_covariance(_make_prices(), lookback_days=252)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10), f"Not PSD; min eigenvalue = {eigvals.min():.2e}"

    def test_diagonal_vols_in_realistic_range(self) -> None:
        """Annualised vols should fall in the 15-30% sector-ETF range."""
        cov = compute_covariance(_make_prices(), lookback_days=252)
        vols = np.sqrt(np.diag(cov))
        assert np.all(vols >= 0.15), f"Vol below 15%; vols={vols.round(3)}"
        assert np.all(vols <= 0.30), f"Vol above 30%; vols={vols.round(3)}"

    def test_raises_on_too_few_rows(self) -> None:
        df = _make_prices(n_days=100)
        with pytest.raises(ValueError, match="needs at least"):
            compute_covariance(df, lookback_days=252)

    def test_raises_exactly_at_boundary(self) -> None:
        # 252 + 1 = 253 rows → should succeed; 252 rows → should fail
        df_ok = _make_prices(n_days=253)
        assert compute_covariance(df_ok, lookback_days=252).shape == (_N, _N)

        df_bad = _make_prices(n_days=252)
        with pytest.raises(ValueError, match="needs at least"):
            compute_covariance(df_bad, lookback_days=252)

    def test_raises_on_nan_prices(self) -> None:
        df = _make_prices()
        df.iloc[10, 3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            compute_covariance(df)

    def test_raises_on_non_positive_prices(self) -> None:
        df = _make_prices()
        df.iloc[5, 0] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            compute_covariance(df)

    def test_sample_method_also_psd(self) -> None:
        cov = compute_covariance(_make_prices(), lookback_days=252, method="sample")
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_lookback_uses_only_recent_rows(self) -> None:
        """High-vol early rows must not pollute the estimate when outside the window."""
        rng = np.random.default_rng(0)
        n_tickers = 5
        high_vol = 0.50 / np.sqrt(252)
        low_vol = 0.15 / np.sqrt(252)

        # 100 high-vol days followed by 250 low-vol days → 350 rows total
        early = rng.standard_normal((100, n_tickers)) * high_vol
        late = rng.standard_normal((250, n_tickers)) * low_vol
        log_ret = np.vstack([early, late])
        prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
        df = pd.DataFrame(prices, columns=[f"E{i}" for i in range(n_tickers)])

        cov_full = compute_covariance(df, lookback_days=349)   # includes early
        cov_recent = compute_covariance(df, lookback_days=249)  # only late

        vol_full = float(np.sqrt(np.diag(cov_full)).mean())
        vol_recent = float(np.sqrt(np.diag(cov_recent)).mean())
        assert vol_recent < vol_full, (
            f"Recent-only vol ({vol_recent:.2%}) should be lower than full-period vol "
            f"({vol_full:.2%}) because early rows had 50% vol."
        )


# ---------------------------------------------------------------------------
# compute_equilibrium_returns
# ---------------------------------------------------------------------------


class TestComputeEquilibriumReturns:
    def test_shape(self) -> None:
        pi = compute_equilibrium_returns(_make_prices(), _EQUAL_WEIGHTS)
        assert pi.shape == (_N,)

    def test_all_positive(self) -> None:
        """With positive-correlated sectors and positive market weights, π > 0."""
        pi = compute_equilibrium_returns(_make_prices(), _EQUAL_WEIGHTS)
        assert np.all(pi > 0), f"Some equilibrium returns non-positive: {pi.round(4)}"

    def test_in_annualized_range(self) -> None:
        """π should be in the 3-12% annualised range for realistic sector-ETF inputs."""
        pi = compute_equilibrium_returns(_make_prices(), _EQUAL_WEIGHTS)
        assert np.all(pi >= 0.03), f"π below 3%: {pi.round(4)}"
        assert np.all(pi <= 0.12), f"π above 12%: {pi.round(4)}"

    def test_ordering_matches_dataframe_columns(self) -> None:
        """π[i] must correspond to prices_df.columns[i], not a sorted order."""
        df = _make_prices(n_tickers=3, n_days=300, seed=1)
        df.columns = ["A", "B", "C"]
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        pi_abc = compute_equilibrium_returns(df, weights)

        df_bca = df[["B", "C", "A"]]
        pi_bca = compute_equilibrium_returns(df_bca, {"B": 0.3, "C": 0.2, "A": 0.5})

        # A is index 0 in pi_abc and index 2 in pi_bca — values must agree
        assert abs(pi_abc[0] - pi_bca[2]) < 1e-12

    def test_weights_renormalized_internally(self) -> None:
        """Unnormalized weights (sum ≠ 1) must give the same result as normalized."""
        df = _make_prices()
        w_raw = {f"ETF_{i}": 5.0 for i in range(_N)}   # sum = 50
        w_norm = {f"ETF_{i}": 0.1 for i in range(_N)}  # sum = 1

        pi_raw = compute_equilibrium_returns(df, w_raw)
        pi_norm = compute_equilibrium_returns(df, w_norm)
        np.testing.assert_allclose(pi_raw, pi_norm, rtol=1e-10)

    def test_scales_linearly_with_risk_aversion(self) -> None:
        """π = λΣw_mkt is linear in λ by definition."""
        df = _make_prices()
        pi_1 = compute_equilibrium_returns(df, _EQUAL_WEIGHTS, risk_aversion=1.0)
        pi_3 = compute_equilibrium_returns(df, _EQUAL_WEIGHTS, risk_aversion=3.0)
        np.testing.assert_allclose(pi_3, 3.0 * pi_1, rtol=1e-10)

    def test_raises_on_missing_ticker(self) -> None:
        df = _make_prices()
        weights = {f"ETF_{i}": 1.0 for i in range(_N - 1)}  # missing ETF_9
        with pytest.raises(ValueError, match="missing tickers"):
            compute_equilibrium_returns(df, weights)

    def test_raises_on_non_positive_risk_aversion(self) -> None:
        df = _make_prices()
        with pytest.raises(ValueError, match="risk_aversion must be positive"):
            compute_equilibrium_returns(df, _EQUAL_WEIGHTS, risk_aversion=0.0)
        with pytest.raises(ValueError, match="risk_aversion must be positive"):
            compute_equilibrium_returns(df, _EQUAL_WEIGHTS, risk_aversion=-2.5)

    def test_raises_on_zero_weights(self) -> None:
        df = _make_prices()
        weights = {f"ETF_{i}": 0.0 for i in range(_N)}
        with pytest.raises(ValueError, match="sum to a positive value"):
            compute_equilibrium_returns(df, weights)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestOptimizerConfigWiring:
    """Verify that optimizer.yaml market_cap_weights round-trips through load_config."""

    def test_config_has_all_ten_etfs(self) -> None:
        from config import load_config
        cfg = load_config("optimizer")
        universe_tickers = [
            "XLK", "XLF", "XLV", "XLY", "XLP",
            "XLE", "XLI", "XLB", "XLRE", "XLU",
        ]
        for ticker in universe_tickers:
            assert ticker in cfg.market_cap_weights, f"{ticker} missing from market_cap_weights"

    def test_config_weights_all_positive(self) -> None:
        from config import load_config
        cfg = load_config("optimizer")
        assert all(w > 0 for w in cfg.market_cap_weights.values())

    def test_config_weights_feed_equilibrium_function(self) -> None:
        """compute_equilibrium_returns accepts raw SSGA weights from config without errors."""
        from config import load_config
        cfg = load_config("optimizer")
        df = _make_prices()
        df.columns = list(cfg.market_cap_weights.keys())  # rename to real tickers
        pi = compute_equilibrium_returns(df, cfg.market_cap_weights, cfg.risk_aversion)
        assert pi.shape == (_N,)
        assert np.all(pi > 0)
