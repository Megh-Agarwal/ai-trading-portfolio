"""Black-Litterman prior: market-implied equilibrium returns and covariance estimation.

Public API:
- compute_covariance: annualised covariance via Ledoit-Wolf shrinkage (or sample).
- compute_equilibrium_returns: implied equilibrium returns π = λ Σ w_mkt.
- get_spy_sector_weights: load and normalise SPY sector weights from config.
- get_prior: orchestrator — loads prices from DB and returns (π, Σ).
"""
from __future__ import annotations

import datetime
import logging
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

if TYPE_CHECKING:
    from sqlalchemy import Engine

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


def get_spy_sector_weights(universe: list[str]) -> np.ndarray:
    """Load SPY sector weights from optimizer.yaml and return as a normalised array.

    Args:
        universe: Ordered list of sector ETF tickers. The returned array follows
            this order so it can be passed directly to compute_equilibrium_returns.

    Returns:
        Weight vector of shape (N,) that sums to 1.0 ± 1e-6.

    Raises:
        ValueError: Any ticker in universe is absent from market_cap_weights config,
            or all config weights are zero or negative.
    """
    from config import load_config

    cfg = load_config("optimizer")
    missing = [t for t in universe if t not in cfg.market_cap_weights]
    if missing:
        raise ValueError(
            f"market_cap_weights in optimizer.yaml is missing tickers: {missing}"
        )
    raw_w = np.array([cfg.market_cap_weights[t] for t in universe], dtype=float)
    if raw_w.sum() <= 0:
        raise ValueError(
            "market_cap_weights entries sum to zero or negative — check optimizer.yaml"
        )
    w = raw_w / raw_w.sum()
    assert abs(w.sum() - 1.0) < 1e-6  # postcondition guard
    return w


def get_prior(
    date: str | datetime.date,
    db: Engine,
) -> tuple[np.ndarray, np.ndarray]:
    """Orchestrate the BL prior: load prices from DB, compute Σ and π.

    Args:
        date: Rebalance date. All price rows up to this date are eligible;
            the most recent `prior.lookback_days` rows (from optimizer.yaml) are used.
        db: SQLAlchemy engine connected to the state database.

    Returns:
        (pi, sigma): equilibrium return vector of shape (N,) and annualised
            covariance matrix of shape (N, N). N = number of tickers in universe.yaml.

    Raises:
        ValueError: Fewer than lookback_days + 1 price rows exist, any ticker is
            missing from market_cap_weights, or no price data found in the DB.
    """
    from config import load_config
    from db.models import Price
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cfg_opt = load_config("optimizer")
    cfg_uni = load_config("universe")
    tickers = cfg_uni.ticker_list
    lookback_days = cfg_opt.prior.lookback_days

    rebalance_date = (
        datetime.date.fromisoformat(date) if isinstance(date, str) else date
    )

    with Session(db) as session:
        rows = session.execute(
            select(Price.date, Price.ticker, Price.adj_close)
            .where(Price.ticker.in_(tickers))
            .where(Price.date <= rebalance_date)
            .order_by(Price.date)
        ).all()

    if not rows:
        raise ValueError(
            f"No price rows found for tickers={tickers} up to date={rebalance_date}. "
            "Run scripts/ingest_prices.py first."
        )

    df_long = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
    prices_df = df_long.pivot(index="date", columns="ticker", values="adj_close")[tickers]
    prices_df.columns.name = None

    w_mkt = get_spy_sector_weights(tickers)
    sigma = compute_covariance(prices_df, lookback_days=lookback_days)
    pi = cfg_opt.risk_aversion * sigma @ w_mkt

    logger.info(
        "get_prior  date=%s  lookback=%d  π=[%.1f%%, %.1f%%]",
        rebalance_date, lookback_days,
        float(pi.min() * 100), float(pi.max() * 100),
    )
    return pi, sigma
