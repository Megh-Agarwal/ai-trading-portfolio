"""Black-Litterman prior: market-implied equilibrium returns and covariance estimation.

Two public functions:
- compute_covariance: annualised covariance via Ledoit-Wolf shrinkage (or sample).
- compute_equilibrium_returns: implied equilibrium returns π = λ Σ w_mkt.

Both functions accept a wide-format prices DataFrame (dates × tickers) and return
plain numpy arrays in ticker-column order. DB queries and DataFrame pivots are the
caller's responsibility so this module stays a pure-math layer.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR: int = 252


def compute_covariance(
    prices_df: pd.DataFrame,
    lookback_days: int = 252,
    method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
) -> np.ndarray:
    """Annualised covariance matrix estimated from historical prices.

    Args:
        prices_df: Wide-format DataFrame — dates as index, tickers as columns,
            values are prices (e.g. adj_close). Must have ≥ lookback_days + 1 rows
            and no NaN or non-positive values.
        lookback_days: Number of most-recent trading days to include (rolling window).
        method: "ledoit_wolf" (default) applies Oracle Approximating Shrinkage for
            a more stable estimate when the number of observations is close to the
            number of assets. "sample" returns raw sample covariance.

    Returns:
        Annualised covariance matrix of shape (N, N), N = len(prices_df.columns).
        Column ordering matches prices_df.columns.

    Raises:
        ValueError: prices_df has fewer than lookback_days + 1 rows, contains NaN,
            or contains non-positive prices (log undefined).
    """
    n_rows, n_assets = prices_df.shape

    if n_rows < lookback_days + 1:
        raise ValueError(
            f"prices_df has {n_rows} rows but needs at least {lookback_days + 1} "
            f"({lookback_days} days of returns + 1 anchor row)."
        )
    if prices_df.isnull().values.any():
        raise ValueError(
            "prices_df contains NaN values; clean the data before calling."
        )
    if (prices_df.values <= 0).any():
        raise ValueError(
            "prices_df contains non-positive prices; log-returns are undefined."
        )

    window = prices_df.iloc[-(lookback_days + 1):]
    log_returns = np.log(window.values[1:] / window.values[:-1])  # shape (T, N)

    if method == "ledoit_wolf":
        daily_cov = LedoitWolf().fit(log_returns).covariance_
    else:
        daily_cov = np.cov(log_returns.T)

    annual_cov = daily_cov * _TRADING_DAYS_PER_YEAR

    vols = np.sqrt(np.diag(annual_cov))
    logger.debug(
        "compute_covariance  method=%s  lookback=%d  n=%d  vol=[%.1f%%, %.1f%%]",
        method, lookback_days, n_assets,
        float(vols.min() * 100), float(vols.max() * 100),
    )
    return annual_cov


def compute_equilibrium_returns(
    prices_df: pd.DataFrame,
    market_cap_weights: dict[str, float],
    risk_aversion: float = 2.5,
    lookback_days: int = 252,
    method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
) -> np.ndarray:
    """Black-Litterman implied equilibrium returns: π = λ Σ w_mkt.

    Args:
        prices_df: Wide-format DataFrame — dates as index, tickers as columns.
        market_cap_weights: Ticker → weight mapping covering every column of prices_df.
            Renormalized to sum=1 internally so raw SSGA percentages can be passed
            directly without manually adjusting for excluded sectors.
        risk_aversion: Market risk-aversion coefficient λ. Default (2.5) matches
            optimizer.yaml; callers should read from config and pass explicitly.
        lookback_days: Passed to compute_covariance.
        method: Passed to compute_covariance.

    Returns:
        Annualised equilibrium return vector π of shape (N,).
        Ordering matches prices_df.columns.

    Raises:
        ValueError: risk_aversion ≤ 0, any ticker in prices_df is missing from
            market_cap_weights, or all weights are zero / negative.
    """
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")

    tickers = list(prices_df.columns)
    missing = [t for t in tickers if t not in market_cap_weights]
    if missing:
        raise ValueError(
            f"market_cap_weights is missing tickers: {missing}. "
            "All columns of prices_df must have a weight entry."
        )

    raw_w = np.array([market_cap_weights[t] for t in tickers], dtype=float)
    if raw_w.sum() <= 0:
        raise ValueError("market_cap_weights must sum to a positive value.")
    w_mkt = raw_w / raw_w.sum()

    sigma = compute_covariance(prices_df, lookback_days=lookback_days, method=method)
    pi = risk_aversion * sigma @ w_mkt

    logger.info(
        "compute_equilibrium_returns  λ=%.1f  n=%d  π=[%.1f%%, %.1f%%]",
        risk_aversion, len(tickers),
        float(pi.min() * 100), float(pi.max() * 100),
    )
    return pi
