"""LLM alpha, per-agent signal attribution, and sector contribution analysis — Ticket 5.6.

Three analysis functions, each independently callable:

  compute_llm_alpha        — weekly and cumulative alpha of backtest_full over backtest_no_llm,
                             with an honest one-sample t-test and power note.
  compute_agent_signal_attribution — Pearson/Spearman correlation of signal strength against
                             next-week realized sector returns, per agent.
  compute_sector_attribution_full  — sector contributions and reconciliation across all weeks,
                             reusing Ticket 4.4's attribution functions with portfolio_id support.

Statistical honesty note: with 53–102 weekly observations, all tests are severely
underpowered for detecting realistic LLM alphas (<2% annualized). Correlation p-values
in the signal attribution assume independence of (week, sector) pairs, which is violated
(returns are correlated across sectors within a week). All functions report these caveats
explicitly in their output.
"""

from __future__ import annotations

import datetime
import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import load_config
from db.models import (
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_LIVE,
    PortfolioSnapshot,
    Price,
    Signal,
)
from eval.attribution import (
    compute_cost_drag,
    compute_period_return,
    compute_sector_contribution,
    reconcile_attribution,
)

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

_RECONCILE_TOL = 0.005  # 0.5% maximum acceptable unexplained fraction
_MDE_Z = 2.802  # z_{0.025} + z_{0.20}: MDE factor for 80% power at 5% two-sided significance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_dated_values(
    portfolio_id: str,
    start: datetime.date,
    end: datetime.date,
    session: Session,
) -> dict[datetime.date, float]:
    """Return {date: total_value} for portfolio_id within [start, end]."""
    rows = session.execute(
        select(PortfolioSnapshot.date, PortfolioSnapshot.total_value)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date >= start)
        .where(PortfolioSnapshot.date <= end)
        .order_by(PortfolioSnapshot.date)
    ).all()
    return {r.date: float(r.total_value) for r in rows}


def _load_sector_prices(
    tickers: list[str],
    dates: list[datetime.date],
    session: Session,
) -> pd.DataFrame:
    """Load adj_close for tickers on the given dates; return DataFrame(index=date_str, cols=ticker).

    Forward-fills from the previous trading day for market holidays (e.g. July 4th)
    that fall on a rebalance Friday. Queries a 7-day buffer before the first date
    so forward-fill always has a prior value to draw from.
    """
    if not dates:
        return pd.DataFrame(index=[], columns=tickers)
    buffer_start = min(dates) - datetime.timedelta(days=7)
    rows = session.execute(
        select(Price.date, Price.ticker, Price.adj_close)
        .where(Price.ticker.in_(tickers))
        .where(Price.date >= buffer_start)
        .where(Price.date <= max(dates))
    ).all()
    df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
    if df.empty:
        return pd.DataFrame(index=[str(d) for d in dates], columns=tickers)
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.reindex(pd.date_range(buffer_start, max(dates), freq="D")).ffill()
    result = pivot.reindex(pd.to_datetime(dates))
    result.index = [str(d) for d in dates]
    result.columns.name = None
    return result.reindex(columns=tickers)


def _corr_metrics(x: list[float], y: list[float]) -> dict:
    """Pearson r, Spearman r, and hit rate for aligned signal and return lists.

    Returns NaN fields when there are fewer than 3 observations.
    """
    n = len(x)
    if n < 3:
        return {
            "n": n,
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "hit_rate": float("nan"),
        }
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    pr, pp = stats.pearsonr(xa, ya)
    sr, sp = stats.spearmanr(xa, ya)
    # Hit rate: sign(signal) == sign(return); exclude zero signals and zero returns
    mask = (xa != 0.0) & (ya != 0.0)
    if mask.any():
        hit_rate = float(np.mean(np.sign(xa[mask]) == np.sign(ya[mask])))
    else:
        hit_rate = float("nan")
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "hit_rate": hit_rate,
    }


# ---------------------------------------------------------------------------
# Public: LLM alpha
# ---------------------------------------------------------------------------


def compute_llm_alpha(
    start_date: str | datetime.date,
    end_date: str | datetime.date,
    db_engine: Engine,
) -> dict:
    """Weekly and cumulative alpha of backtest_full over backtest_no_llm.

    Alpha is defined as r_full[t] − r_nollm[t] for each week t where both
    portfolios have a snapshot.  Cumulative alpha is the log of
    (1+r_full) / (1+r_nollm) running product.

    Statistical test: one-sample two-sided t-test of weekly alpha against zero.
    The power note states the minimum detectable effect at 80% power, 5%
    significance given the observed alpha volatility and sample size.

    Args:
        start_date: Start of evaluation window (inclusive).
        end_date: End of evaluation window (inclusive).
        db_engine: SQLAlchemy Engine connected to state.db.

    Returns:
        Dict with weekly_alpha, cumulative_alpha, t_statistic, p_value,
        mean_alpha_weekly, std_alpha_weekly, and power_note.
    """
    start = datetime.date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = datetime.date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    with Session(db_engine) as session:
        full_vals = _load_dated_values(PORTFOLIO_BACKTEST_FULL, start, end, session)
        nollm_vals = _load_dated_values(PORTFOLIO_BACKTEST_NO_LLM, start, end, session)

    # Align dates (inner join)
    shared_dates = sorted(set(full_vals) & set(nollm_vals))

    if len(shared_dates) < 2:
        logger.warning("compute_llm_alpha: fewer than 2 aligned dates in [%s, %s]", start, end)
        return {
            "n_weeks": 0,
            "weekly_alpha": [],
            "cumulative_alpha": [],
            "mean_alpha_weekly": float("nan"),
            "std_alpha_weekly": float("nan"),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "statistically_significant": False,
            "min_detectable_alpha_weekly": float("nan"),
            "min_detectable_alpha_annualized": float("nan"),
            "power_note": "Insufficient data for statistical test.",
        }

    # Build aligned value arrays
    full_v = np.array([full_vals[d] for d in shared_dates])
    nollm_v = np.array([nollm_vals[d] for d in shared_dates])

    # Weekly simple returns (len = n_dates - 1)
    r_full = full_v[1:] / full_v[:-1] - 1.0
    r_nollm = nollm_v[1:] / nollm_v[:-1] - 1.0
    weekly_alpha = r_full - r_nollm
    n_weeks = len(weekly_alpha)

    # Cumulative alpha: product of (1 + r_full) / product of (1 + r_nollm) - 1
    cum_full = np.cumprod(1.0 + r_full)
    cum_nollm = np.cumprod(1.0 + r_nollm)
    cum_alpha = cum_full / cum_nollm - 1.0

    mean_alpha = float(np.mean(weekly_alpha))
    std_alpha = float(np.std(weekly_alpha, ddof=1)) if n_weeks >= 2 else float("nan")

    # One-sample t-test (requires n >= 2 for std estimation)
    if n_weeks >= 2:
        t_stat, p_val = stats.ttest_1samp(weekly_alpha, popmean=0.0)
        t_stat, p_val = float(t_stat), float(p_val)
    else:
        t_stat, p_val = float("nan"), float("nan")
    significant = p_val < 0.05 if not math.isnan(p_val) else False

    # Minimum detectable effect at 80% power, 5% two-sided significance
    if not math.isnan(std_alpha):
        mde_weekly = _MDE_Z * std_alpha / math.sqrt(n_weeks)
    else:
        mde_weekly = float("nan")
    mde_annual = mde_weekly * 52 if not math.isnan(mde_weekly) else float("nan")

    power_note = (
        f"With {n_weeks} weekly observations and σ_alpha={std_alpha * 10000:.1f} bps/week, "
        f"this test has 80% power to detect a true mean alpha ≥ {mde_weekly * 10000:.1f} bps/week "
        f"({mde_annual * 100:.1f}% annualized) at α=0.05 two-sided. "
        "Realistic LLM alphas are likely far smaller; treat the t-test as indicative only."
    )

    # Build week-by-week output (one entry per return period)
    return_dates = shared_dates[1:]  # return[i] = period from dates[i-1] to dates[i]
    weekly_rows = [
        {
            "date": str(d),
            "alpha": float(a),
            "r_full": float(rf),
            "r_nollm": float(rn),
            "cumulative_alpha": float(ca),
        }
        for d, a, rf, rn, ca in zip(return_dates, weekly_alpha, r_full, r_nollm, cum_alpha)
    ]

    return {
        "n_weeks": n_weeks,
        "weekly_alpha": weekly_rows,
        "cumulative_alpha_final": float(cum_alpha[-1]),
        "mean_alpha_weekly": mean_alpha,
        "mean_alpha_annualized": mean_alpha * 52,
        "std_alpha_weekly": std_alpha,
        "t_statistic": t_stat,
        "p_value": p_val,
        "statistically_significant": significant,
        "min_detectable_alpha_weekly": mde_weekly,
        "min_detectable_alpha_annualized": mde_annual,
        "power_note": power_note,
    }


# ---------------------------------------------------------------------------
# Public: per-agent signal attribution
# ---------------------------------------------------------------------------


def compute_agent_signal_attribution(
    start_date: str | datetime.date,
    end_date: str | datetime.date,
    db_engine: Engine,
    portfolio_id: str = PORTFOLIO_BACKTEST_FULL,
) -> dict:
    """Correlate each agent's signal strength against the subsequent week's realized returns.

    For news and polymarket: signal at date t vs sector ETF return from t to t+1.
    For macro: macro_regime signal at date t vs portfolio return from t to t+1.

    Metrics per agent: Pearson r, Spearman r, hit rate (sign match fraction), and
    conviction-weighted variants (signal × confidence).

    Args:
        start_date: Start of evaluation window (inclusive).
        end_date: End of evaluation window (inclusive).
        db_engine: SQLAlchemy Engine connected to state.db.
        portfolio_id: Portfolio whose signals to analyse (default: backtest_full).

    Returns:
        Dict keyed by agent name ("news", "macro", "polymarket") containing
        correlation metrics, plus an interpretation_note.
    """
    start = datetime.date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = datetime.date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    tickers = load_config("universe").ticker_list

    with Session(db_engine) as session:
        # Load all signals for this portfolio in range
        signal_rows = session.execute(
            select(Signal.date, Signal.agent_name, Signal.target, Signal.signal_value,
                   Signal.confidence)
            .where(Signal.portfolio_id == portfolio_id)
            .where(Signal.date >= start)
            .where(Signal.date <= end)
            .order_by(Signal.date)
        ).all()

        if not signal_rows:
            logger.warning(
                "compute_agent_signal_attribution: no signals for portfolio=%s", portfolio_id
            )
            empty = {
                "n": 0,
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_r": float("nan"),
                "spearman_p": float("nan"),
                "hit_rate": float("nan"),
                "weighted_pearson_r": float("nan"),
                "weighted_spearman_r": float("nan"),
            }
            return {
                "news": empty,
                "macro": empty,
                "polymarket": empty,
                "interpretation_note": "No signals found in DB for this portfolio and date range.",
            }

        # All signal dates, sorted
        sig_dates = sorted({r.date for r in signal_rows})

        # Load portfolio values for macro attribution
        port_vals = _load_dated_values(portfolio_id, start, end, session)

        # Load sector prices for news / polymarket
        price_dates = sorted({d for d in port_vals})
        prices_df = _load_sector_prices(tickers, price_dates, session)

    # Group signals by (agent_name, date, target)
    # signals_map[agent_name][date][target] = (signal_value, confidence)
    signals_map: dict[str, dict[datetime.date, dict[str, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in signal_rows:
        conf = float(r.confidence) if r.confidence is not None else 0.0
        signals_map[r.agent_name][r.date][r.target] = (float(r.signal_value), conf)

    # Build (signal, next_return, confidence) tuples per agent
    news_sv, news_ret, news_conf = [], [], []
    poly_sv, poly_ret, poly_conf = [], [], []
    macro_sv, macro_ret, macro_conf = [], [], []

    port_dates = sorted(port_vals)

    for i, sig_date in enumerate(sig_dates):
        # Find next available portfolio date after sig_date
        next_dates = [d for d in port_dates if d > sig_date]
        if not next_dates:
            continue
        next_date = next_dates[0]

        # Portfolio return for macro
        if sig_date in port_vals and next_date in port_vals and port_vals[sig_date] > 0:
            port_return = port_vals[next_date] / port_vals[sig_date] - 1.0
        else:
            port_return = None

        # Sector prices
        sig_str = str(sig_date)
        next_str = str(next_date)

        # News signals
        if "news" in signals_map and sig_date in signals_map["news"]:
            for sector, (sv, conf) in signals_map["news"][sig_date].items():
                if sector in (prices_df.columns if prices_df is not None else []):
                    try:
                        p0 = float(prices_df.loc[sig_str, sector])
                        p1 = float(prices_df.loc[next_str, sector])
                        if not (math.isnan(p0) or math.isnan(p1)) and p0 > 0:
                            news_sv.append(sv)
                            news_ret.append(p1 / p0 - 1.0)
                            news_conf.append(conf)
                    except (KeyError, TypeError):
                        pass

        # Polymarket signals
        if "polymarket" in signals_map and sig_date in signals_map["polymarket"]:
            for sector, (sv, conf) in signals_map["polymarket"][sig_date].items():
                if sector in (prices_df.columns if prices_df is not None else []):
                    try:
                        p0 = float(prices_df.loc[sig_str, sector])
                        p1 = float(prices_df.loc[next_str, sector])
                        if not (math.isnan(p0) or math.isnan(p1)) and p0 > 0:
                            poly_sv.append(sv)
                            poly_ret.append(p1 / p0 - 1.0)
                            poly_conf.append(conf)
                    except (KeyError, TypeError):
                        pass

        # Macro: macro_regime target vs portfolio return
        if "macro" in signals_map and sig_date in signals_map["macro"]:
            if port_return is not None and "macro_regime" in signals_map["macro"][sig_date]:
                sv, conf = signals_map["macro"][sig_date]["macro_regime"]
                macro_sv.append(sv)
                macro_ret.append(port_return)
                macro_conf.append(conf)

    def _agent_result(sv_list: list, ret_list: list, conf_list: list) -> dict:
        base = _corr_metrics(sv_list, ret_list)
        # Conviction-weighted signals
        if len(sv_list) >= 3 and conf_list:
            weighted = [s * c for s, c in zip(sv_list, conf_list)]
            w_base = _corr_metrics(weighted, ret_list)
            base["weighted_pearson_r"] = w_base["pearson_r"]
            base["weighted_spearman_r"] = w_base["spearman_r"]
        else:
            base["weighted_pearson_r"] = float("nan")
            base["weighted_spearman_r"] = float("nan")
        return base

    n_obs_news = len(news_sv)
    n_obs_macro = len(macro_sv)

    interpretation_note = (
        f"News/polymarket correlations use {n_obs_news} (week, sector) pairs; "
        f"macro uses {n_obs_macro} weekly portfolio returns. "
        "Observations within a week are cross-sectionally correlated, so "
        "p-values substantially overstate significance — treat as exploratory only."
    )

    return {
        "news": _agent_result(news_sv, news_ret, news_conf),
        "macro": _agent_result(macro_sv, macro_ret, macro_conf),
        "polymarket": _agent_result(poly_sv, poly_ret, poly_conf),
        "interpretation_note": interpretation_note,
    }


# ---------------------------------------------------------------------------
# Public: sector attribution across all weeks
# ---------------------------------------------------------------------------


def compute_sector_attribution_full(
    portfolio_id: str = PORTFOLIO_LIVE,
    start_date: str | datetime.date | None = None,
    end_date: str | datetime.date | None = None,
    db_engine: Engine | None = None,
) -> dict:
    """Run Ticket 4.4 attribution functions across every consecutive week pair.

    For each pair of consecutive snapshot dates: compute per-sector contributions,
    cost drag, total return, and reconciliation. Aggregate contributions by sector.
    The final reconciliation checks that the overall unexplained fraction is within
    _RECONCILE_TOL = 0.5%.

    Args:
        portfolio_id: Portfolio namespace for all DB queries.
        start_date: Start of evaluation window (inclusive).
        end_date: End of evaluation window (inclusive).
        db_engine: SQLAlchemy Engine connected to state.db.

    Returns:
        Dict with per-sector aggregated contributions, per-week detail, total cost drag,
        and reconciliation result including a "reconciled" boolean.
    """
    start = datetime.date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = datetime.date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    tickers = load_config("universe").ticker_list

    with Session(db_engine) as session:
        # Get all snapshot dates for this portfolio in range, sorted
        snap_dates = session.execute(
            select(PortfolioSnapshot.date)
            .where(PortfolioSnapshot.portfolio_id == portfolio_id)
            .where(PortfolioSnapshot.date >= start)
            .where(PortfolioSnapshot.date <= end)
            .order_by(PortfolioSnapshot.date)
        ).scalars().all()
        snap_dates = list(snap_dates)

        if len(snap_dates) < 2:
            logger.warning(
                "compute_sector_attribution_full: fewer than 2 snapshots for portfolio=%s",
                portfolio_id,
            )
            return {
                "portfolio_id": portfolio_id,
                "n_weeks": 0,
                "sector_contributions_total": {},
                "weeks": [],
                "total_cost_drag_bps": 0.0,
                "reconciliation": {"reconciled": False, "unexplained_pct": float("nan")},
            }

        prices_df = _load_sector_prices(tickers, snap_dates, session)

        week_results: list[dict] = []
        sector_totals: dict[str, float] = defaultdict(float)
        total_cost_drag_bps = 0.0
        sum_weekly_returns = 0.0

        for i in range(len(snap_dates) - 1):
            d_start = snap_dates[i]
            d_end = snap_dates[i + 1]
            ds = str(d_start)
            de = str(d_end)

            try:
                period = compute_period_return(ds, de, session, portfolio_id=portfolio_id)
                contribs = compute_sector_contribution(
                    ds, de, session, prices_df, portfolio_id=portfolio_id
                )
                drag = compute_cost_drag(ds, de, session, portfolio_id=portfolio_id)
                recon = reconcile_attribution(
                    period["total_return_pct"], contribs, drag["cost_drag_bps"]
                )

                for sector, c in contribs.items():
                    sector_totals[sector] += c
                total_cost_drag_bps += drag["cost_drag_bps"]
                sum_weekly_returns += period["total_return_pct"]

                week_results.append(
                    {
                        "date_start": ds,
                        "date_end": de,
                        "total_return_pct": period["total_return_pct"],
                        "sector_contributions": contribs,
                        "cost_drag_bps": drag["cost_drag_bps"],
                        "unexplained_pct": recon["unexplained_pct"],
                        "week_reconciled": abs(recon["unexplained_pct"]) < _RECONCILE_TOL,
                    }
                )

            except (ValueError, KeyError) as exc:
                logger.warning("Week %s→%s: attribution skipped (%s)", ds, de, exc)

    # Overall reconciliation: sum of weekly returns vs sum of contributions minus cost drag
    sum_contribs = sum(sector_totals.values())
    total_cost_frac = total_cost_drag_bps / 10_000.0
    explained = sum_contribs - total_cost_frac
    unexplained = sum_weekly_returns - explained
    all_weeks_reconciled = all(w["week_reconciled"] for w in week_results)

    return {
        "portfolio_id": portfolio_id,
        "n_weeks": len(week_results),
        "sector_contributions_total": dict(sector_totals),
        "weeks": week_results,
        "total_cost_drag_bps": total_cost_drag_bps,
        "reconciliation": {
            "sum_weekly_returns": sum_weekly_returns,
            "sum_contributions": sum_contribs,
            "total_cost_drag_fraction": total_cost_frac,
            "explained": explained,
            "unexplained_pct": unexplained,
            "all_weeks_within_tolerance": all_weeks_reconciled,
            "tolerance": _RECONCILE_TOL,
            "reconciled": abs(unexplained) < _RECONCILE_TOL and all_weeks_reconciled,
        },
    }


# ---------------------------------------------------------------------------
# Public: combined entry point
# ---------------------------------------------------------------------------


def compute_all_analysis(
    start_date: str | datetime.date,
    end_date: str | datetime.date,
    db_engine: Engine,
    attribution_portfolio_id: str = PORTFOLIO_BACKTEST_FULL,
) -> dict:
    """Run all three analyses and return a combined result dict.

    Args:
        start_date: Start of evaluation window (inclusive).
        end_date: End of evaluation window (inclusive).
        db_engine: SQLAlchemy Engine.
        attribution_portfolio_id: Portfolio for sector attribution.

    Returns:
        Dict with keys "llm_alpha", "signal_attribution", "sector_attribution".
    """
    return {
        "llm_alpha": compute_llm_alpha(start_date, end_date, db_engine),
        "signal_attribution": compute_agent_signal_attribution(start_date, end_date, db_engine),
        "sector_attribution": compute_sector_attribution_full(
            portfolio_id=attribution_portfolio_id,
            start_date=start_date,
            end_date=end_date,
            db_engine=db_engine,
        ),
    }
