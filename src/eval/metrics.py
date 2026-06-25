"""Portfolio performance metrics — Ticket 5.5.

All metric functions are pure (take numpy arrays) for easy testing.
DB-backed loaders (_load_*) are private; the public entry point is
compute_all_metrics which opens its own session.

Annualisation convention: 52 weeks per year throughout.
Bootstrap CI uses a basic block bootstrap (overlapping start positions)
to preserve short-range autocorrelation in weekly returns.
Downside deviation in Sortino uses the population formula (divide by n),
which is standard in finance.
"""

from __future__ import annotations

import datetime
import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import PORTFOLIO_LIVE, PortfolioSnapshot, TargetWeight, Trade

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

_WEEKS_PER_YEAR: int = 52


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


def sharpe_ratio(weekly_returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio from weekly returns (rf=0 default).

    Args:
        weekly_returns: 1-D array of weekly simple returns (e.g. 0.01 = 1%).
        risk_free_annual: Annual risk-free rate (decimal). Converted to weekly.

    Returns:
        Annualised Sharpe, or nan when fewer than 2 observations or zero vol.
    """
    r = np.asarray(weekly_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    rf_w = risk_free_annual / _WEEKS_PER_YEAR
    excess = r - rf_w
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return float("nan")
    return float(np.mean(excess) / std * math.sqrt(_WEEKS_PER_YEAR))


def sortino_ratio(weekly_returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    """Annualised Sortino ratio from weekly returns.

    Downside deviation = sqrt(mean(min(r - rf_w, 0)^2)) — population formula,
    including all weeks in the denominator (not just negative ones).

    Returns:
        Annualised Sortino, nan when < 2 obs, inf when no weeks below rf.
    """
    r = np.asarray(weekly_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    rf_w = risk_free_annual / _WEEKS_PER_YEAR
    excess = r - rf_w
    downside_var = float(np.mean(np.minimum(excess, 0.0) ** 2))
    if downside_var == 0.0:
        return float("inf")
    return float(np.mean(excess) / math.sqrt(downside_var) * math.sqrt(_WEEKS_PER_YEAR))


def max_drawdown(total_values: np.ndarray) -> float:
    """Worst peak-to-trough decline as a decimal fraction (always <= 0).

    Args:
        total_values: Chronological portfolio values (at least 2 required).

    Returns:
        Worst drawdown (e.g. -0.15 = −15%). Returns 0.0 for < 2 values.
    """
    v = np.asarray(total_values, dtype=float)
    if len(v) < 2:
        return 0.0
    running_max = np.maximum.accumulate(v)
    drawdowns = v / running_max - 1.0
    return float(np.min(drawdowns))


def total_return(total_values: np.ndarray) -> float:
    """Compound total return: (V_end / V_start) - 1.

    Returns nan when fewer than 2 values or starting value is zero.
    """
    v = np.asarray(total_values, dtype=float)
    if len(v) < 2 or v[0] == 0.0:
        return float("nan")
    return float(v[-1] / v[0] - 1.0)


def annualized_vol(weekly_returns: np.ndarray) -> float:
    """Annualised volatility of weekly returns: std(r, ddof=1) * sqrt(52).

    Returns nan when fewer than 2 observations.
    """
    r = np.asarray(weekly_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * math.sqrt(_WEEKS_PER_YEAR))


def calmar_ratio(
    weekly_returns: np.ndarray,
    total_values: np.ndarray,
    risk_free_annual: float = 0.0,
) -> float:
    """Calmar ratio: annualised excess return / abs(max drawdown).

    Annualised return is computed via CAGR from total return and n_weeks.

    Returns nan when max drawdown is zero or total return is nan.
    """
    tr = total_return(total_values)
    md = max_drawdown(total_values)
    n_weeks = len(weekly_returns)
    if math.isnan(tr) or md == 0.0 or n_weeks == 0:
        return float("nan")
    ann_return = (1.0 + tr) ** (_WEEKS_PER_YEAR / n_weeks) - 1.0
    excess_ann = ann_return - risk_free_annual
    return float(excess_ann / abs(md))


def avg_weekly_turnover(turnovers: np.ndarray) -> float:
    """Mean one-way weekly turnover (L1 norm of weight changes).

    Args:
        turnovers: Per-week turnover values (same scale as compute_turnover).

    Returns:
        Mean turnover, or 0.0 for an empty array.
    """
    t = np.asarray(turnovers, dtype=float)
    return float(np.mean(t)) if len(t) > 0 else 0.0


def bootstrap_sharpe_ci(
    weekly_returns: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.90,
    block_size: int = 4,
    rng_seed: int | None = None,
) -> dict:
    """Block-bootstrap confidence interval for the Sharpe ratio.

    Uses overlapping blocks of length block_size to preserve autocorrelation.
    ceil(n / block_size) blocks are sampled with replacement per resample,
    concatenated, then trimmed to length n.

    Args:
        weekly_returns: 1-D array of weekly simple returns.
        n_resamples: Number of bootstrap resamples (>= 1000 recommended).
        ci: Confidence level (e.g. 0.90 → 90% CI).
        block_size: Block length in weeks. Default 4 ≈ monthly.
        rng_seed: Seed for reproducibility (None for random state).

    Returns:
        Dict with keys lo, hi, width (all floats).
        lo and hi are nan when there are too few observations.
    """
    r = np.asarray(weekly_returns, dtype=float)
    n = len(r)
    if n < block_size + 1:
        return {"lo": float("nan"), "hi": float("nan"), "width": float("nan")}

    rng = np.random.default_rng(rng_seed)
    alpha = (1.0 - ci) / 2.0
    n_blocks = math.ceil(n / block_size)
    max_start = n - block_size

    sharpes: list[float] = []
    for _ in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        resampled = np.concatenate([r[s : s + block_size] for s in starts])[:n]
        s = sharpe_ratio(resampled)
        if not math.isnan(s):
            sharpes.append(s)

    if not sharpes:
        return {"lo": float("nan"), "hi": float("nan"), "width": float("nan")}

    arr = np.array(sharpes, dtype=float)
    lo = float(np.percentile(arr, 100.0 * alpha))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha)))
    return {"lo": lo, "hi": hi, "width": hi - lo}


# ---------------------------------------------------------------------------
# DB-backed loaders (private — called from compute_all_metrics)
# ---------------------------------------------------------------------------


def _load_snapshot_values(
    portfolio_id: str,
    start: datetime.date,
    end: datetime.date,
    session: Session,
) -> np.ndarray:
    """Return chronological total_value array for portfolio_id in [start, end]."""
    rows = session.execute(
        select(PortfolioSnapshot.total_value)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .where(PortfolioSnapshot.date >= start)
        .where(PortfolioSnapshot.date <= end)
        .order_by(PortfolioSnapshot.date)
    ).scalars().all()
    return np.array(rows, dtype=float)


def _load_weekly_returns(
    portfolio_id: str,
    start: datetime.date,
    end: datetime.date,
    session: Session,
) -> np.ndarray:
    """Derive weekly simple returns from consecutive snapshot total_values."""
    values = _load_snapshot_values(portfolio_id, start, end, session)
    if len(values) < 2:
        return np.array([], dtype=float)
    return values[1:] / values[:-1] - 1.0


def _load_weekly_turnover(
    portfolio_id: str,
    start: datetime.date,
    end: datetime.date,
    session: Session,
) -> np.ndarray:
    """Return per-week L1 turnover from consecutive TargetWeight snapshots.

    Turnover = sum(|w_new[i] - w_old[i]|) for all sectors between weeks.
    """
    rows = session.execute(
        select(TargetWeight.date, TargetWeight.sector, TargetWeight.weight)
        .where(TargetWeight.portfolio_id == portfolio_id)
        .where(TargetWeight.date >= start)
        .where(TargetWeight.date <= end)
        .order_by(TargetWeight.date)
    ).all()

    weights_by_date: dict[datetime.date, dict[str, float]] = defaultdict(dict)
    ordered_dates: list[datetime.date] = []
    for row in rows:
        if row.date not in weights_by_date:
            ordered_dates.append(row.date)
        weights_by_date[row.date][row.sector] = float(row.weight)

    if len(ordered_dates) < 2:
        return np.array([], dtype=float)

    all_sectors: set[str] = set()
    for d in ordered_dates:
        all_sectors |= weights_by_date[d].keys()

    turnovers: list[float] = []
    for i in range(1, len(ordered_dates)):
        prev = weights_by_date[ordered_dates[i - 1]]
        curr = weights_by_date[ordered_dates[i]]
        to = sum(abs(curr.get(s, 0.0) - prev.get(s, 0.0)) for s in all_sectors)
        turnovers.append(to)

    return np.array(turnovers, dtype=float)


def _total_cost_drag_bps(
    portfolio_id: str,
    start: datetime.date,
    end: datetime.date,
    session: Session,
) -> float:
    """Total commission as basis points of average portfolio value.

    Returns 0.0 when there are no trades or no snapshots in the period.
    """
    total_commission = session.execute(
        select(func.sum(Trade.commission))
        .where(Trade.portfolio_id == portfolio_id)
        .where(Trade.date >= start)
        .where(Trade.date <= end)
    ).scalar()
    if total_commission is None or total_commission == 0.0:
        return 0.0

    values = _load_snapshot_values(portfolio_id, start, end, session)
    if len(values) == 0:
        return 0.0
    avg_value = float(np.mean(values))
    if avg_value == 0.0:
        return 0.0
    return float(total_commission / avg_value * 10_000.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    portfolio_id: str = PORTFOLIO_LIVE,
    start_date: str | datetime.date | None = None,
    end_date: str | datetime.date | None = None,
    db_engine: Engine | None = None,
    n_bootstrap: int = 1000,
    block_size: int = 4,
    rng_seed: int | None = None,
) -> dict:
    """Compute the full performance metric suite for one portfolio over a date range.

    Metrics include Sharpe, Sortino, max drawdown, Calmar, total return,
    annualised volatility, average/total turnover, transaction cost drag,
    and a block-bootstrap 90% CI for the Sharpe ratio.

    The bootstrap CI width is always reported and flagged in bootstrap_note
    to ensure it is never hidden from the reader.

    Args:
        portfolio_id: Portfolio namespace (scopes all DB queries).
        start_date: Start of evaluation window (inclusive). ISO string or date.
        end_date: End of evaluation window (inclusive). ISO string or date.
        db_engine: SQLAlchemy Engine connected to state.db.
        n_bootstrap: Number of bootstrap resamples (>= 1000 recommended).
        block_size: Block length in weeks for the bootstrap.
        rng_seed: RNG seed for reproducible bootstrap CIs.

    Returns:
        Dict with all metrics plus bootstrap_note describing the CI width caveat.
    """
    start = datetime.date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = datetime.date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    with Session(db_engine) as session:
        values = _load_snapshot_values(portfolio_id, start, end, session)
        wr = _load_weekly_returns(portfolio_id, start, end, session)
        turnovers = _load_weekly_turnover(portfolio_id, start, end, session)
        cost_drag = _total_cost_drag_bps(portfolio_id, start, end, session)

    n_weeks = len(wr)
    if n_weeks == 0:
        logger.warning("No weekly returns for portfolio=%s in [%s, %s]", portfolio_id, start, end)

    sr = sharpe_ratio(wr)
    so = sortino_ratio(wr)
    md = max_drawdown(values)
    tr = total_return(values)
    av = annualized_vol(wr)
    ca = calmar_ratio(wr, values)
    awt = avg_weekly_turnover(turnovers)
    total_to = float(np.sum(turnovers)) if len(turnovers) > 0 else 0.0

    ci = bootstrap_sharpe_ci(wr, n_resamples=n_bootstrap, block_size=block_size, rng_seed=rng_seed)

    logger.info(
        "compute_all_metrics  portfolio=%s  n_weeks=%d  sharpe=%.3f  sortino=%.3f  "
        "max_dd=%.2f%%  total_return=%.2f%%  ci_width=%.3f",
        portfolio_id,
        n_weeks,
        sr if not math.isnan(sr) else 0.0,
        so if not (math.isnan(so) or math.isinf(so)) else 0.0,
        md * 100,
        tr * 100 if not math.isnan(tr) else 0.0,
        ci["width"] if not math.isnan(ci["width"]) else 0.0,
    )

    return {
        "portfolio_id": portfolio_id,
        "start_date": str(start),
        "end_date": str(end),
        "n_weeks": n_weeks,
        "total_return": tr,
        "annualized_vol": av,
        "sharpe_ratio": sr,
        "sortino_ratio": so,
        "max_drawdown": md,
        "calmar_ratio": ca,
        "avg_weekly_turnover": awt,
        "total_turnover": total_to,
        "total_cost_drag_bps": cost_drag,
        "sharpe_bootstrap_ci_90": (ci["lo"], ci["hi"]),
        "sharpe_bootstrap_ci_width": ci["width"],
        "bootstrap_n_resamples": n_bootstrap,
        "bootstrap_block_size": block_size,
        "bootstrap_note": (
            f"Bootstrap Sharpe 90% CI width = {ci['width']:.3f} based on {n_weeks} weekly "
            "observations. A minimum of ~5 years (260+ weeks) is needed for stable Sharpe "
            "inference; treat this CI as indicative only."
        ),
    }
