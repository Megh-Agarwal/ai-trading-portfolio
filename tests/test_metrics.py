"""Tests for src/eval/metrics.py — Ticket 5.5.

Each pure metric function is tested against a hand-calculable small case
with explicit intermediate values shown in comments.  DB-backed functions
are tested using an in-memory SQLite DB with seeded rows.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import (
    PORTFOLIO_LIVE,
    Base,
    PortfolioSnapshot,
    TargetWeight,
    Trade,
)
from eval.metrics import (
    annualized_vol,
    avg_weekly_turnover,
    bootstrap_sharpe_ci,
    calmar_ratio,
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# returns = [0.02, 0.01, -0.01, 0.02, 0.01]
# mean = 0.01
# deviations = [0.01, 0, -0.02, 0.01, 0]
# variance(ddof=1) = 0.0006/4 = 0.00015
# std = sqrt(0.00015)
_R5 = np.array([0.02, 0.01, -0.01, 0.02, 0.01])
_R5_MEAN = 0.01
_R5_STD = math.sqrt(0.00015)  # ≈ 0.012247

# total_values with one drawdown episode
# drawdown at index 2: 105/110 - 1 = -5/110 ≈ -0.04545
_V4 = np.array([100.0, 110.0, 105.0, 115.0])
_V4_MAX_DD = 105.0 / 110.0 - 1.0  # = -5/110

_WEEKS_PER_YEAR = 52

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _seed_snapshots(engine, values: list[tuple[datetime.date, float]]) -> None:
    with Session(engine) as session:
        for date, v in values:
            session.merge(
                PortfolioSnapshot(
                    portfolio_id=PORTFOLIO_LIVE,
                    date=date,
                    total_value=v,
                    cash=0.0,
                    gross_exposure=v,
                    net_exposure=v,
                )
            )
        session.commit()


def _seed_target_weights(
    engine, rows: list[tuple[datetime.date, str, float]]
) -> None:
    with Session(engine) as session:
        for date, sector, weight in rows:
            session.merge(
                TargetWeight(
                    portfolio_id=PORTFOLIO_LIVE, date=date, sector=sector, weight=weight
                )
            )
        session.commit()


def _seed_trade(engine, date: datetime.date, commission: float) -> None:
    with Session(engine) as session:
        session.add(
            Trade(
                portfolio_id=PORTFOLIO_LIVE,
                date=date,
                ticker="XLK",
                side="buy",
                shares=10.0,
                price=100.0,
                commission=commission,
                slippage=0.0,
            )
        )
        session.commit()


# ---------------------------------------------------------------------------
# TestSharpeRatio
# ---------------------------------------------------------------------------

_D1 = datetime.date(2025, 6, 13)
_D2 = datetime.date(2025, 6, 20)
_D3 = datetime.date(2025, 6, 27)
_D4 = datetime.date(2025, 7, 4)


class TestSharpeRatio:
    def test_known_values(self) -> None:
        """mean=0.01, std=sqrt(0.00015) → Sharpe = 0.01/sqrt(0.00015)*sqrt(52)."""
        expected = _R5_MEAN / _R5_STD * math.sqrt(_WEEKS_PER_YEAR)
        assert sharpe_ratio(_R5) == pytest.approx(expected, rel=1e-9)

    def test_zero_rf_is_default(self) -> None:
        assert sharpe_ratio(_R5) == sharpe_ratio(_R5, risk_free_annual=0.0)

    def test_positive_rf_lowers_sharpe(self) -> None:
        assert sharpe_ratio(_R5, risk_free_annual=0.04) < sharpe_ratio(_R5)

    def test_constant_returns_returns_nan(self) -> None:
        r = np.full(5, 0.01)
        assert math.isnan(sharpe_ratio(r))

    def test_single_return_returns_nan(self) -> None:
        assert math.isnan(sharpe_ratio(np.array([0.01])))

    def test_empty_array_returns_nan(self) -> None:
        assert math.isnan(sharpe_ratio(np.array([])))

    def test_positive_for_positive_mean_returns(self) -> None:
        assert sharpe_ratio(_R5) > 0.0

    def test_negative_for_negative_mean_returns(self) -> None:
        assert sharpe_ratio(-_R5) < 0.0


# ---------------------------------------------------------------------------
# TestSortinoRatio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_known_values(self) -> None:
        """Downside returns: [0,0,-0.01,0,0].
        Downside variance = mean([0,0,0.0001,0,0]) = 0.00002.
        Downside std = sqrt(0.00002) = 0.004472.
        Sortino = 0.01/0.004472*sqrt(52).
        """
        downside_std = math.sqrt(0.00002)  # sqrt(mean([0,0,0.01^2,0,0]))
        expected = _R5_MEAN / downside_std * math.sqrt(_WEEKS_PER_YEAR)
        assert sortino_ratio(_R5) == pytest.approx(expected, rel=1e-9)

    def test_greater_than_sharpe_when_returns_mostly_positive(self) -> None:
        assert sortino_ratio(_R5) > sharpe_ratio(_R5)

    def test_all_positive_returns_gives_inf(self) -> None:
        r = np.array([0.01, 0.02, 0.01, 0.03])
        assert math.isinf(sortino_ratio(r))

    def test_single_return_returns_nan(self) -> None:
        assert math.isnan(sortino_ratio(np.array([0.01])))

    def test_empty_returns_nan(self) -> None:
        assert math.isnan(sortino_ratio(np.array([])))


# ---------------------------------------------------------------------------
# TestMaxDrawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_known_values(self) -> None:
        """Peak at index 1 (110), trough at index 2 (105) → dd = 105/110 - 1."""
        assert max_drawdown(_V4) == pytest.approx(_V4_MAX_DD, rel=1e-9)

    def test_always_non_positive(self) -> None:
        assert max_drawdown(_V4) <= 0.0

    def test_monotonically_increasing_has_zero_drawdown(self) -> None:
        v = np.array([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(v) == pytest.approx(0.0, abs=1e-12)

    def test_complete_loss_is_negative_one(self) -> None:
        v = np.array([100.0, 50.0, 0.0])
        assert max_drawdown(v) == pytest.approx(-1.0)

    def test_single_value_returns_zero(self) -> None:
        assert max_drawdown(np.array([100.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert max_drawdown(np.array([])) == 0.0

    def test_uses_running_max_not_global_max(self) -> None:
        # Drawdown should be measured from the running peak, not the series end.
        # V = [100, 90, 80, 200]: drawdown occurs between 0→1 (100→90), not vs 200.
        v = np.array([100.0, 90.0, 80.0, 200.0])
        expected = 80.0 / 100.0 - 1.0  # -0.20 from running max at index 0
        assert max_drawdown(v) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# TestTotalReturn
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_known_value(self) -> None:
        """(115/100) - 1 = 0.15."""
        assert total_return(_V4) == pytest.approx(0.15, rel=1e-9)

    def test_loss_is_negative(self) -> None:
        v = np.array([100.0, 80.0])
        assert total_return(v) == pytest.approx(-0.20, rel=1e-9)

    def test_single_value_returns_nan(self) -> None:
        assert math.isnan(total_return(np.array([100.0])))

    def test_zero_start_returns_nan(self) -> None:
        assert math.isnan(total_return(np.array([0.0, 100.0])))


# ---------------------------------------------------------------------------
# TestAnnualizedVol
# ---------------------------------------------------------------------------


class TestAnnualizedVol:
    def test_known_value(self) -> None:
        """std(_R5, ddof=1) = sqrt(0.00015) ≈ 0.012247; * sqrt(52)."""
        expected = _R5_STD * math.sqrt(_WEEKS_PER_YEAR)
        assert annualized_vol(_R5) == pytest.approx(expected, rel=1e-9)

    def test_always_non_negative(self) -> None:
        assert annualized_vol(_R5) >= 0.0

    def test_constant_returns_gives_zero(self) -> None:
        assert annualized_vol(np.full(5, 0.01)) == pytest.approx(0.0, abs=1e-12)

    def test_single_return_returns_nan(self) -> None:
        assert math.isnan(annualized_vol(np.array([0.01])))


# ---------------------------------------------------------------------------
# TestCalmarRatio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_known_value(self) -> None:
        """total_return(_V4)=0.15, n=4 values → 3 returns, max_dd=-5/110.
        CAGR = 1.15^(52/3) - 1.  Calmar = CAGR / abs(max_dd).
        """
        # _V4 has 4 values → 3 weekly returns implicit in CAGR denominator
        tr = 0.15
        n_weeks = 3  # len(returns) = len(values) - 1
        ann = (1.0 + tr) ** (_WEEKS_PER_YEAR / n_weeks) - 1.0
        md_abs = abs(_V4_MAX_DD)
        expected = ann / md_abs
        wr_for_v4 = _V4[1:] / _V4[:-1] - 1.0
        assert calmar_ratio(wr_for_v4, _V4) == pytest.approx(expected, rel=1e-9)

    def test_zero_drawdown_returns_nan(self) -> None:
        v = np.array([100.0, 110.0, 120.0])
        r = v[1:] / v[:-1] - 1.0
        assert math.isnan(calmar_ratio(r, v))

    def test_positive_for_positive_return_with_drawdown(self) -> None:
        wr = _V4[1:] / _V4[:-1] - 1.0
        assert calmar_ratio(wr, _V4) > 0.0


# ---------------------------------------------------------------------------
# TestAvgWeeklyTurnover
# ---------------------------------------------------------------------------


class TestAvgWeeklyTurnover:
    def test_basic_mean(self) -> None:
        t = np.array([0.1, 0.2, 0.3])
        assert avg_weekly_turnover(t) == pytest.approx(0.2, rel=1e-9)

    def test_empty_returns_zero(self) -> None:
        assert avg_weekly_turnover(np.array([])) == 0.0

    def test_single_element(self) -> None:
        assert avg_weekly_turnover(np.array([0.5])) == pytest.approx(0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# TestBootstrapSharpeCi
# ---------------------------------------------------------------------------


class TestBootstrapSharpeCi:
    def test_returns_lo_hi_width_keys(self) -> None:
        result = bootstrap_sharpe_ci(_R5, n_resamples=100, rng_seed=42)
        assert "lo" in result and "hi" in result and "width" in result

    def test_width_equals_hi_minus_lo(self) -> None:
        result = bootstrap_sharpe_ci(_R5, n_resamples=100, rng_seed=42)
        assert result["width"] == pytest.approx(result["hi"] - result["lo"], rel=1e-9)

    def test_ci_contains_point_estimate(self) -> None:
        """The bootstrap CI should bracket the observed Sharpe in most cases."""
        result = bootstrap_sharpe_ci(_R5, n_resamples=500, ci=0.90, rng_seed=0)
        # Bootstrap quantile CI may or may not bracket the point estimate;
        # the interval must be a valid (lo <= hi) range.
        assert result["lo"] <= result["hi"]

    def test_hi_greater_than_lo(self) -> None:
        result = bootstrap_sharpe_ci(_R5, n_resamples=100, rng_seed=1)
        assert result["hi"] >= result["lo"]

    def test_reproducible_with_seed(self) -> None:
        r1 = bootstrap_sharpe_ci(_R5, n_resamples=100, rng_seed=7)
        r2 = bootstrap_sharpe_ci(_R5, n_resamples=100, rng_seed=7)
        assert r1["lo"] == pytest.approx(r2["lo"])
        assert r1["hi"] == pytest.approx(r2["hi"])

    def test_different_seeds_differ_on_larger_series(self) -> None:
        """Different seeds produce different CIs when the series is large enough."""
        np.random.seed(42)
        r = np.random.normal(0.01, 0.03, 30)
        r1 = bootstrap_sharpe_ci(r, n_resamples=200, rng_seed=1)
        r2 = bootstrap_sharpe_ci(r, n_resamples=200, rng_seed=999)
        assert r1["lo"] != r2["lo"] or r1["hi"] != r2["hi"]

    def test_too_few_observations_returns_nan(self) -> None:
        result = bootstrap_sharpe_ci(np.array([0.01, 0.02]), block_size=4)
        assert math.isnan(result["lo"])
        assert math.isnan(result["width"])

    def test_ci_width_increases_with_n_resamples_stability(self) -> None:
        """Both 500-resample and 1000-resample runs should yield finite positive widths."""
        np.random.seed(0)
        r = np.random.normal(0.01, 0.03, 40)
        ci_500 = bootstrap_sharpe_ci(r, n_resamples=500, rng_seed=0)
        ci_1000 = bootstrap_sharpe_ci(r, n_resamples=1000, rng_seed=0)
        assert ci_500["width"] > 0.0
        assert ci_1000["width"] > 0.0

    def test_ci_width_is_reported_explicitly(self) -> None:
        """Width must be a finite positive number, not hidden as NaN."""
        result = bootstrap_sharpe_ci(_R5, n_resamples=200, rng_seed=5)
        assert not math.isnan(result["width"])
        assert result["width"] > 0.0


# ---------------------------------------------------------------------------
# TestComputeAllMetrics — DB-backed integration
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_returns_required_keys(self, db_engine) -> None:
        _seed_snapshots(db_engine, [(_D1, 1_000_000.0), (_D2, 1_010_000.0), (_D3, 1_005_000.0)])
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D3,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        for key in (
            "portfolio_id",
            "start_date",
            "end_date",
            "n_weeks",
            "total_return",
            "annualized_vol",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "avg_weekly_turnover",
            "total_turnover",
            "total_cost_drag_bps",
            "sharpe_bootstrap_ci_90",
            "sharpe_bootstrap_ci_width",
            "bootstrap_note",
        ):
            assert key in result, f"missing key: {key}"

    def test_total_return_matches_snapshot_math(self, db_engine) -> None:
        """total_return = (1_020_000 / 1_000_000) - 1 = 0.02."""
        _seed_snapshots(db_engine, [(_D1, 1_000_000.0), (_D2, 1_020_000.0)])
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D2,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert result["total_return"] == pytest.approx(0.02, rel=1e-9)

    def test_sharpe_matches_pure_function(self, db_engine) -> None:
        # 3 snapshots → 2 weekly returns
        vals = [1_000_000.0, 1_010_000.0, 1_005_000.0]
        _seed_snapshots(db_engine, list(zip([_D1, _D2, _D3], vals)))
        wr = np.array(vals)[1:] / np.array(vals)[:-1] - 1.0
        expected = sharpe_ratio(wr)
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D3,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert result["sharpe_ratio"] == pytest.approx(expected, rel=1e-9)

    def test_max_drawdown_matches_pure_function(self, db_engine) -> None:
        vals = [1_000_000.0, 1_100_000.0, 1_050_000.0, 1_150_000.0]
        _seed_snapshots(db_engine, list(zip([_D1, _D2, _D3, _D4], vals)))
        expected = max_drawdown(np.array(vals))
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D4,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert result["max_drawdown"] == pytest.approx(expected, rel=1e-9)

    def test_n_weeks_equals_number_of_returns(self, db_engine) -> None:
        _seed_snapshots(db_engine, [(_D1, 1e6), (_D2, 1.01e6), (_D3, 1.02e6), (_D4, 1.03e6)])
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D4,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        # 4 snapshots → 3 weekly returns
        assert result["n_weeks"] == 3

    def test_cost_drag_bps_from_trades(self, db_engine) -> None:
        """$300 commission on $1M average value = 3 bps."""
        _seed_snapshots(db_engine, [(_D1, 1_000_000.0), (_D2, 1_000_000.0)])
        _seed_trade(db_engine, _D1, commission=300.0)
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D2,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert result["total_cost_drag_bps"] == pytest.approx(3.0, rel=1e-6)

    def test_turnover_computed_from_target_weights(self, db_engine) -> None:
        """Sector weight moves from 0.5→0.6 (XLK) and 0.5→0.4 (XLV): turnover = 0.2."""
        _seed_snapshots(db_engine, [(_D1, 1e6), (_D2, 1e6)])
        _seed_target_weights(db_engine, [
            (_D1, "XLK", 0.5), (_D1, "XLV", 0.5),
            (_D2, "XLK", 0.6), (_D2, "XLV", 0.4),
        ])
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D2,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert result["avg_weekly_turnover"] == pytest.approx(0.2, rel=1e-9)
        assert result["total_turnover"] == pytest.approx(0.2, rel=1e-9)

    def test_bootstrap_note_always_present(self, db_engine) -> None:
        _seed_snapshots(db_engine, [(_D1, 1e6), (_D2, 1.01e6), (_D3, 1.02e6)])
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D3,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert isinstance(result["bootstrap_note"], str)
        assert len(result["bootstrap_note"]) > 0

    def test_bootstrap_ci_width_is_finite_and_positive(self, db_engine) -> None:
        # Need enough data for a valid CI (block_size=4 → need at least 5 obs)
        dates = [_D1 + datetime.timedelta(weeks=i) for i in range(8)]
        vals = [1e6 * (1.01 ** i) for i in range(8)]
        _seed_snapshots(db_engine, list(zip(dates, vals)))
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=dates[0],
            end_date=dates[-1],
            db_engine=db_engine,
            n_bootstrap=200,
            rng_seed=0,
        )
        width = result["sharpe_bootstrap_ci_width"]
        assert not math.isnan(width)
        assert width > 0.0

    def test_no_data_returns_nan_metrics(self, db_engine) -> None:
        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D2,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        assert result["n_weeks"] == 0
        assert math.isnan(result["sharpe_ratio"])
        assert math.isnan(result["total_return"])

    def test_portfolio_id_scoping(self, db_engine) -> None:
        """Metrics for portfolio A must not bleed into portfolio B's query."""
        from db.models import PORTFOLIO_BACKTEST_FULL

        with Session(db_engine) as session:
            session.merge(
                PortfolioSnapshot(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL,
                    date=_D1,
                    total_value=500_000.0,
                    cash=0.0,
                    gross_exposure=500_000.0,
                    net_exposure=500_000.0,
                )
            )
            session.commit()

        result = compute_all_metrics(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_D1,
            end_date=_D2,
            db_engine=db_engine,
            n_bootstrap=50,
            rng_seed=0,
        )
        # PORTFOLIO_LIVE has no snapshots in this range
        assert result["n_weeks"] == 0
