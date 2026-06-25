"""Tests for src/eval/analysis.py — Ticket 5.6.

All tests use in-memory SQLite with seeded rows.  No real agent calls.
Hand-calculable values are shown in comments beside each assertion.
"""

from __future__ import annotations

import datetime
import math

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import (
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_LIVE,
    Base,
    PortfolioSnapshot,
    Position,
    Price,
    Signal,
    Trade,
)
from eval.analysis import (
    compute_agent_signal_attribution,
    compute_all_analysis,
    compute_llm_alpha,
    compute_sector_attribution_full,
)

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_W1 = datetime.date(2025, 6, 13)
_W2 = datetime.date(2025, 6, 20)
_W3 = datetime.date(2025, 6, 27)
_W4 = datetime.date(2025, 7, 4)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _snap(engine, portfolio_id: str, date: datetime.date, total_value: float) -> None:
    with Session(engine) as s:
        s.merge(
            PortfolioSnapshot(
                portfolio_id=portfolio_id,
                date=date,
                total_value=total_value,
                cash=0.0,
                gross_exposure=total_value,
                net_exposure=total_value,
            )
        )
        s.commit()


def _price(engine, date: datetime.date, ticker: str, adj_close: float) -> None:
    with Session(engine) as s:
        s.merge(
            Price(
                date=date,
                ticker=ticker,
                open=adj_close,
                high=adj_close,
                low=adj_close,
                close=adj_close,
                volume=1_000_000,
                adj_close=adj_close,
            )
        )
        s.commit()


def _signal(
    engine,
    portfolio_id: str,
    date: datetime.date,
    agent_name: str,
    target: str,
    signal_value: float,
    confidence: float = 0.5,
) -> None:
    with Session(engine) as s:
        s.add(
            Signal(
                portfolio_id=portfolio_id,
                date=date,
                agent_name=agent_name,
                target=target,
                signal_value=signal_value,
                confidence=confidence,
            )
        )
        s.commit()


def _position(
    engine,
    portfolio_id: str,
    date: datetime.date,
    ticker: str,
    shares: float,
    market_value: float,
) -> None:
    with Session(engine) as s:
        s.merge(
            Position(
                portfolio_id=portfolio_id,
                date=date,
                ticker=ticker,
                shares=shares,
                market_value=market_value,
                cost_basis=market_value,
            )
        )
        s.commit()


def _trade(engine, portfolio_id: str, date: datetime.date, commission: float) -> None:
    with Session(engine) as s:
        s.add(
            Trade(
                portfolio_id=portfolio_id,
                date=date,
                ticker="XLK",
                side="buy",
                shares=1.0,
                price=100.0,
                commission=commission,
                slippage=0.0,
            )
        )
        s.commit()


# ---------------------------------------------------------------------------
# TestComputeLlmAlpha
# ---------------------------------------------------------------------------


class TestComputeLlmAlpha:
    def test_returns_required_keys(self, db_engine) -> None:
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_010_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W2, 1_005_000.0)

        result = compute_llm_alpha(_W1, _W2, db_engine)
        for key in (
            "n_weeks",
            "weekly_alpha",
            "cumulative_alpha_final",
            "mean_alpha_weekly",
            "std_alpha_weekly",
            "t_statistic",
            "p_value",
            "statistically_significant",
            "power_note",
        ):
            assert key in result, f"missing key: {key}"

    def test_positive_alpha_when_full_beats_nollm(self, db_engine) -> None:
        """full: +1%, nollm: +0.5% → alpha = +0.5%."""
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_010_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W2, 1_005_000.0)

        result = compute_llm_alpha(_W1, _W2, db_engine)
        # r_full = 0.01, r_nollm = 0.005 → alpha = 0.005
        assert result["n_weeks"] == 1
        assert result["weekly_alpha"][0]["alpha"] == pytest.approx(0.005, rel=1e-9)
        assert result["mean_alpha_weekly"] == pytest.approx(0.005, rel=1e-9)

    def test_negative_alpha_when_full_trails_nollm(self, db_engine) -> None:
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_005_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W2, 1_010_000.0)

        result = compute_llm_alpha(_W1, _W2, db_engine)
        assert result["mean_alpha_weekly"] < 0.0

    def test_zero_alpha_when_returns_identical(self, db_engine) -> None:
        for pid in (PORTFOLIO_BACKTEST_FULL, PORTFOLIO_BACKTEST_NO_LLM):
            _snap(db_engine, pid, _W1, 1_000_000.0)
            _snap(db_engine, pid, _W2, 1_010_000.0)
            _snap(db_engine, pid, _W3, 1_020_000.0)

        result = compute_llm_alpha(_W1, _W3, db_engine)
        assert result["mean_alpha_weekly"] == pytest.approx(0.0, abs=1e-12)

    def test_cumulative_alpha_correct(self, db_engine) -> None:
        """full: +2%, +1% | nollm: +1%, +2%.
        cum_full = 1.02 × 1.01 = 1.0302
        cum_nollm = 1.01 × 1.02 = 1.0302 → cumulative alpha = 0 (symmetric).
        """
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_020_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W3, 1_030_200.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W2, 1_010_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W3, 1_030_200.0)

        result = compute_llm_alpha(_W1, _W3, db_engine)
        assert result["cumulative_alpha_final"] == pytest.approx(0.0, abs=1e-9)

    def test_handles_misaligned_dates(self, db_engine) -> None:
        """Only dates present in BOTH portfolios count."""
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_010_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W3, 1_020_000.0)  # not in no_llm
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_NO_LLM, _W2, 1_005_000.0)

        result = compute_llm_alpha(_W1, _W3, db_engine)
        # Only W1 and W2 are shared → 1 weekly return
        assert result["n_weeks"] == 1

    def test_insufficient_data_returns_nan_fields(self, db_engine) -> None:
        result = compute_llm_alpha(_W1, _W2, db_engine)
        assert result["n_weeks"] == 0
        assert math.isnan(result["t_statistic"])

    def test_power_note_always_present(self, db_engine) -> None:
        for pid in (PORTFOLIO_BACKTEST_FULL, PORTFOLIO_BACKTEST_NO_LLM):
            _snap(db_engine, pid, _W1, 1_000_000.0)
            _snap(db_engine, pid, _W2, 1_010_000.0)
            _snap(db_engine, pid, _W3, 1_005_000.0)
        result = compute_llm_alpha(_W1, _W3, db_engine)
        assert isinstance(result["power_note"], str)
        assert len(result["power_note"]) > 50

    def test_mean_alpha_annualized_is_weekly_times_52(self, db_engine) -> None:
        for pid in (PORTFOLIO_BACKTEST_FULL, PORTFOLIO_BACKTEST_NO_LLM):
            _snap(db_engine, pid, _W1, 1_000_000.0)
            _snap(db_engine, pid, _W2, 1_010_000.0)
            _snap(db_engine, pid, _W3, 1_005_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_015_000.0)  # overwrite to create gap

        result = compute_llm_alpha(_W1, _W3, db_engine)
        assert result["mean_alpha_annualized"] == pytest.approx(
            result["mean_alpha_weekly"] * 52, rel=1e-9
        )


# ---------------------------------------------------------------------------
# TestComputeAgentSignalAttribution
# ---------------------------------------------------------------------------


class TestComputeAgentSignalAttribution:
    def _seed_news_scenario(self, engine) -> None:
        """4 weeks: news signal → next-week XLK return, perfect positive correlation."""
        # Signals: W1=+1.0, W2=-1.0, W3=+0.5, W4=-0.5 (W4 signal has no next week)
        # XLK prices: W1=100, W2=102, W3=101, W4=101.5
        # Returns: W1→W2: +2%, W2→W3: -0.98%, W3→W4: +0.50%
        # Signal at W1 predicts W2 return → (1.0, +0.02)
        # Signal at W2 predicts W3 return → (-1.0, -0.0098...)
        # Signal at W3 predicts W4 return → (0.5, +0.005...)
        # Pearson corr([1, -1, 0.5], [0.02, -0.0098, 0.005]) > 0
        for date, sv in [(_W1, 1.0), (_W2, -1.0), (_W3, 0.5)]:
            _signal(engine, PORTFOLIO_BACKTEST_FULL, date, "news", "XLK", sv, confidence=0.8)
        for date, p in [(_W1, 100.0), (_W2, 102.0), (_W3, 101.0), (_W4, 101.5)]:
            _price(engine, date, "XLK", p)
        for date, v in [(_W1, 1e6), (_W2, 1.02e6), (_W3, 1.01e6), (_W4, 1.015e6)]:
            _snap(engine, PORTFOLIO_BACKTEST_FULL, date, v)

    def test_returns_required_keys(self, db_engine) -> None:
        self._seed_news_scenario(db_engine)
        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        for key in ("news", "macro", "polymarket", "interpretation_note"):
            assert key in result
        for sub_key in ("n", "pearson_r", "spearman_r", "hit_rate"):
            assert sub_key in result["news"]

    def test_positive_correlation_for_aligned_news_signals(self, db_engine) -> None:
        self._seed_news_scenario(db_engine)
        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        # Signals and returns are positively aligned → Pearson r > 0
        assert result["news"]["pearson_r"] > 0.0

    def test_hit_rate_all_correct(self, db_engine) -> None:
        """All 3 signals point in the same direction as the subsequent return."""
        self._seed_news_scenario(db_engine)
        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        assert result["news"]["hit_rate"] == pytest.approx(1.0, rel=1e-9)

    def test_returns_nan_when_no_signals(self, db_engine) -> None:
        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        assert result["news"]["n"] == 0
        assert math.isnan(result["news"]["pearson_r"])

    def test_interpretation_note_always_present(self, db_engine) -> None:
        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        assert isinstance(result["interpretation_note"], str)
        assert len(result["interpretation_note"]) > 10

    def test_macro_signal_uses_portfolio_return(self, db_engine) -> None:
        """Macro signal predicts portfolio return (not sector return)."""
        # macro_regime=+1 on W1, portfolio: W1=1M, W2=1.02M (+2%)
        _signal(
            db_engine, PORTFOLIO_BACKTEST_FULL, _W1, "macro", "macro_regime", 1.0, confidence=0.7
        )
        _signal(
            db_engine, PORTFOLIO_BACKTEST_FULL, _W2, "macro", "macro_regime", -1.0, confidence=0.7
        )
        _signal(
            db_engine, PORTFOLIO_BACKTEST_FULL, _W3, "macro", "macro_regime", 1.0, confidence=0.7
        )
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W1, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W2, 1_020_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W3, 1_000_000.0)
        _snap(db_engine, PORTFOLIO_BACKTEST_FULL, _W4, 1_020_000.0)

        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        assert result["macro"]["n"] >= 1

    def test_weighted_pearson_r_is_reported(self, db_engine) -> None:
        self._seed_news_scenario(db_engine)
        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        assert "weighted_pearson_r" in result["news"]
        # With n=3, weighted metric should also be computable
        assert not math.isnan(result["news"]["weighted_pearson_r"])

    def test_negative_correlation_for_reversed_signals(self, db_engine) -> None:
        """Signals reversed relative to returns → Pearson r < 0."""
        for date, sv in [(_W1, -1.0), (_W2, 1.0), (_W3, -0.5)]:
            _signal(db_engine, PORTFOLIO_BACKTEST_FULL, date, "news", "XLK", sv, confidence=0.8)
        for date, p in [(_W1, 100.0), (_W2, 102.0), (_W3, 101.0), (_W4, 101.5)]:
            _price(db_engine, date, "XLK", p)
        for date, v in [(_W1, 1e6), (_W2, 1.02e6), (_W3, 1.01e6), (_W4, 1.015e6)]:
            _snap(db_engine, PORTFOLIO_BACKTEST_FULL, date, v)

        result = compute_agent_signal_attribution(_W1, _W4, db_engine)
        assert result["news"]["pearson_r"] < 0.0


# ---------------------------------------------------------------------------
# TestComputeSectorAttributionFull
# ---------------------------------------------------------------------------


class TestComputeSectorAttributionFull:
    def _seed_two_week_scenario(self, engine) -> None:
        """Two weeks; XLK: 100→105 (+5%), XLF: 50→50 (0%).
        Position: 100 XLK shares + 0 XLF each week.
        w_xlk_start = 100*100 / 10000 = 1.0 (100% XLK, no cash/other)
        w_xlk_end   = 100*105 / 10500 = 1.0
        avg_w_xlk = 1.0
        contribution_xlk = 1.0 × 0.05 = 0.05
        total_return = 10500 / 10000 - 1 = 0.05
        unexplained ≈ 0 (no cost drag, perfect coverage)
        """
        # Snapshots
        for pid in (PORTFOLIO_LIVE,):
            _snap(engine, pid, _W1, 10_000.0)
            _snap(engine, pid, _W2, 10_500.0)

        # Prices
        _price(engine, _W1, "XLK", 100.0)
        _price(engine, _W2, "XLK", 105.0)

        # Positions
        _position(engine, PORTFOLIO_LIVE, _W1, "XLK", 100.0, 10_000.0)
        _position(engine, PORTFOLIO_LIVE, _W2, "XLK", 100.0, 10_500.0)
        _position(engine, PORTFOLIO_LIVE, _W1, "CASH", 0.0, 0.0)
        _position(engine, PORTFOLIO_LIVE, _W2, "CASH", 0.0, 0.0)

    def test_returns_required_keys(self, db_engine) -> None:
        self._seed_two_week_scenario(db_engine)
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        for key in (
            "portfolio_id",
            "n_weeks",
            "sector_contributions_total",
            "weeks",
            "total_cost_drag_bps",
            "reconciliation",
        ):
            assert key in result

    def test_sector_contribution_correct(self, db_engine) -> None:
        """XLK contribution = avg_weight (1.0) × sector_return (0.05) = 0.05."""
        self._seed_two_week_scenario(db_engine)
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        xlk_contrib = result["sector_contributions_total"].get("XLK", 0.0)
        assert xlk_contrib == pytest.approx(0.05, rel=1e-6)

    def test_reconciliation_within_tolerance(self, db_engine) -> None:
        """No cost drag + full coverage → unexplained ≈ 0."""
        self._seed_two_week_scenario(db_engine)
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        recon = result["reconciliation"]
        assert abs(recon["unexplained_pct"]) < 0.005  # within 0.5%
        assert recon["reconciled"] is True

    def test_cost_drag_included_in_reconciliation(self, db_engine) -> None:
        """Adding a $50 trade commission on $10k portfolio = 5bps of drag."""
        self._seed_two_week_scenario(db_engine)
        _trade(db_engine, PORTFOLIO_LIVE, _W1, commission=50.0)
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        # avg_value = (10k + 10.5k) / 2 = 10.25k; drag = 50/10250 × 10000 ≈ 48.78 bps
        assert result["total_cost_drag_bps"] == pytest.approx(50.0 / 10_250.0 * 10_000, rel=1e-4)

    def test_portfolio_id_scoped(self, db_engine) -> None:
        """Seeding data for PORTFOLIO_LIVE must not affect PORTFOLIO_BACKTEST_FULL."""
        self._seed_two_week_scenario(db_engine)
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_BACKTEST_FULL,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        assert result["n_weeks"] == 0

    def test_no_data_returns_gracefully(self, db_engine) -> None:
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        assert result["n_weeks"] == 0
        assert result["reconciliation"]["reconciled"] is False

    def test_n_weeks_matches_consecutive_snapshot_pairs(self, db_engine) -> None:
        for date, v in [(_W1, 1e6), (_W2, 1.01e6), (_W3, 1.02e6)]:
            _snap(db_engine, PORTFOLIO_LIVE, date, v)
        for date, p in [(_W1, 100.0), (_W2, 101.0), (_W3, 102.0)]:
            _price(db_engine, date, "XLK", p)
        for date, mv in [(_W1, 1e6), (_W2, 1.01e6), (_W3, 1.02e6)]:
            _position(db_engine, PORTFOLIO_LIVE, date, "XLK", 10_000.0, mv)
            _position(db_engine, PORTFOLIO_LIVE, date, "CASH", 0.0, 0.0)

        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W3,
            db_engine=db_engine,
        )
        # 3 snapshots → 2 consecutive pairs → n_weeks = 2
        assert result["n_weeks"] == 2

    def test_reconciliation_key_in_output(self, db_engine) -> None:
        self._seed_two_week_scenario(db_engine)
        result = compute_sector_attribution_full(
            portfolio_id=PORTFOLIO_LIVE,
            start_date=_W1,
            end_date=_W2,
            db_engine=db_engine,
        )
        for key in ("sum_weekly_returns", "sum_contributions", "unexplained_pct", "reconciled"):
            assert key in result["reconciliation"]


# ---------------------------------------------------------------------------
# TestComputeAllAnalysis
# ---------------------------------------------------------------------------


class TestComputeAllAnalysis:
    def test_returns_three_top_level_keys(self, db_engine) -> None:
        result = compute_all_analysis(_W1, _W2, db_engine)
        assert "llm_alpha" in result
        assert "signal_attribution" in result
        assert "sector_attribution" in result
