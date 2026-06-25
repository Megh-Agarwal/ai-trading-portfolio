"""Integration tests for src/weekly_run.py — Ticket 4.5.

All three agent calls (NewsAgent, MacroAgent, PolymarketAgent) are mocked via
a patch on agents.pipeline.run_agent_pipeline — no real Claude API calls are
made and no LLM costs are incurred.  Ingestion is similarly mocked.

The optimizer pipeline is mocked with a side_effect that writes TargetWeight
rows to the test DB, mirroring what the real optimizer does, so that
downstream execution steps (generate_orders, simulate_fills, apply_fills)
exercise the real code paths.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from config import load_config
from db.models import (
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_NO_LLM,
    Base,
    PortfolioSnapshot,
    Position,
    TargetWeight,
    Trade,
)
from weekly_run import _already_executed, _fetch_prices_for_date, run_weekly

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_DATE = "2024-06-07"
_DATE_OBJ = datetime.date(2024, 6, 7)
_TICKERS = load_config("universe").ticker_list
_N = len(_TICKERS)
# 97% invested — 3% held as cash buffer to cover transaction costs on first run.
# At 100% invested from a 100% cash portfolio, 3bps costs exceed floor-rounding savings.
_EQUAL_WEIGHTS = {t: 0.097 for t in _TICKERS}

# Realistic fake prices for all 10 sector ETFs
_FAKE_PRICES = {
    "XLK": 200.0,
    "XLF": 40.0,
    "XLV": 130.0,
    "XLY": 170.0,
    "XLP": 75.0,
    "XLE": 85.0,
    "XLI": 110.0,
    "XLB": 90.0,
    "XLRE": 40.0,
    "XLU": 65.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    """In-memory SQLite engine with all tables and fake price rows for _DATE."""
    from db.models import Price

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Seed prices for the rebalance date so _fetch_prices_for_date returns data
    with Session(engine) as session:
        for ticker, price in _FAKE_PRICES.items():
            session.add(
                Price(
                    date=_DATE_OBJ,
                    ticker=ticker,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=1_000_000,
                    adj_close=price,
                )
            )
        session.commit()

    yield engine
    engine.dispose()


def _make_optimizer_mock():
    """side_effect for run_optimization_pipeline: writes TargetWeight rows + returns dict."""

    def _mock(date, db, mode="backtest", portfolio_id="live", force_zero_views=False):
        date_obj = datetime.date.fromisoformat(date) if isinstance(date, str) else date
        with Session(db) as session:
            session.execute(
                delete(TargetWeight)
                .where(TargetWeight.portfolio_id == portfolio_id)
                .where(TargetWeight.date == date_obj)
            )
            for t, w in _EQUAL_WEIGHTS.items():
                session.add(
                    TargetWeight(portfolio_id=portfolio_id, date=date_obj, sector=t, weight=w)
                )
            session.commit()
        return {
            "date": str(date_obj),
            "weights": _EQUAL_WEIGHTS.copy(),
            "expected_return_annual": 0.08,
            "expected_vol_annual": 0.12,
            "turnover": 0.10,
            "estimated_cost_usd": 300.0,
            "risk_checks": [],
            "any_risk_triggered": False,
            "mode": mode,
            "views_available": True,
            "vol_constraint_status": "not_binding",
        }

    return _mock


def _make_agent_mock():
    """side_effect for run_agent_pipeline — returns canned dict, no API calls."""

    def _mock(date, db, mode="backtest", weights=None, portfolio_id="live"):
        return {
            "date": str(date),
            "signals_by_agent": {
                "sentiment": {"status": "ok"},
                "macro": {"status": "ok"},
                "events": {"status": "ok"},
            },
            "views": {"q": [0.0] * _N, "omega_diag": [1.0] * _N},
            "total_cost_usd": 0.0,
            "total_latency_ms": 1.0,
        }

    return _mock


def _make_equal_weight_mock():
    """side_effect for run_equal_weight_pipeline: writes equal TargetWeight rows."""

    def _mock(date, db, portfolio_id="live"):
        date_obj = datetime.date.fromisoformat(date) if isinstance(date, str) else date
        equal_w = 1.0 / _N
        with Session(db) as session:
            session.execute(
                delete(TargetWeight)
                .where(TargetWeight.portfolio_id == portfolio_id)
                .where(TargetWeight.date == date_obj)
            )
            for t in _TICKERS:
                session.add(
                    TargetWeight(portfolio_id=portfolio_id, date=date_obj, sector=t, weight=equal_w)
                )
            session.commit()
        return {
            "date": str(date_obj),
            "weights": {t: equal_w for t in _TICKERS},
            "expected_return_annual": 0.0,
            "expected_vol_annual": 0.0,
            "turnover": 0.0,
            "estimated_cost_usd": 0.0,
            "risk_checks": [],
            "any_risk_triggered": False,
            "mode": "equal_weight",
            "views_available": False,
            "vol_constraint_status": "not_applicable",
        }

    return _mock


# ---------------------------------------------------------------------------
# Core patches applied to every test in this module
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_all_external(monkeypatch):
    """Patch ingestion, agents, and optimizer so tests run with $0 API cost."""
    with (
        patch("weekly_run._ingest_fresh_data", return_value=None),
        patch("weekly_run.run_agent_pipeline", side_effect=_make_agent_mock()),
        patch("weekly_run.run_optimization_pipeline", side_effect=_make_optimizer_mock()),
    ):
        yield


# ---------------------------------------------------------------------------
# Mandatory acceptance-criteria tests
# ---------------------------------------------------------------------------


class TestFullSequenceWritesExpectedRows:
    def test_positions_written_after_run(self, db_engine) -> None:
        """run_weekly writes at least one Position row for the rebalance date."""
        run_weekly(_DATE, "backtest", db_engine)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count()).select_from(Position).where(Position.date == _DATE_OBJ)
            ).scalar()
        assert count > 0

    def test_snapshot_written_after_run(self, db_engine) -> None:
        """run_weekly writes exactly one PortfolioSnapshot row for the date."""
        run_weekly(_DATE, "backtest", db_engine)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(PortfolioSnapshot)
                .where(PortfolioSnapshot.date == _DATE_OBJ)
            ).scalar()
        assert count == 1

    def test_summary_keys_present(self, db_engine) -> None:
        result = run_weekly(_DATE, "backtest", db_engine)
        for key in ("date", "mode", "llm_cost_usd", "weights_after", "ending_portfolio_value"):
            assert key in result, f"missing key: {key}"

    def test_summary_skipped_false_on_first_run(self, db_engine) -> None:
        result = run_weekly(_DATE, "backtest", db_engine)
        assert result.get("skipped") is False

    def test_target_weights_written_to_db(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count()).select_from(TargetWeight).where(TargetWeight.date == _DATE_OBJ)
            ).scalar()
        assert count == _N


class TestIdempotencySkipsSecondRun:
    def test_rerrun_without_force_is_noop(self, db_engine) -> None:
        """Second run without --force returns skipped=True and adds no new trade rows."""
        run_weekly(_DATE, "backtest", db_engine)

        with Session(db_engine) as session:
            trade_count_after_first = session.execute(
                select(func.count()).select_from(Trade).where(Trade.date == _DATE_OBJ)
            ).scalar()

        result2 = run_weekly(_DATE, "backtest", db_engine)

        with Session(db_engine) as session:
            trade_count_after_second = session.execute(
                select(func.count()).select_from(Trade).where(Trade.date == _DATE_OBJ)
            ).scalar()

        assert result2.get("skipped") is True
        assert trade_count_after_second == trade_count_after_first

    def test_skipped_result_contains_date(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine)
        result2 = run_weekly(_DATE, "backtest", db_engine)
        assert result2["date"] == _DATE
        assert "already executed" in result2["reason"].lower()


class TestForceFlagReExecutes:
    def test_force_reruns_despite_existing_data(self, db_engine) -> None:
        """--force bypasses the idempotency check and runs again."""
        run_weekly(_DATE, "backtest", db_engine)
        result2 = run_weekly(_DATE, "backtest", db_engine, force=True)
        assert result2.get("skipped") is not True

    def test_force_writes_new_snapshot(self, db_engine) -> None:
        """Snapshot is overwritten (merge semantics) on forced re-run."""
        run_weekly(_DATE, "backtest", db_engine)
        run_weekly(_DATE, "backtest", db_engine, force=True)

        # Exactly one snapshot (merge is idempotent on primary key)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(PortfolioSnapshot)
                .where(PortfolioSnapshot.date == _DATE_OBJ)
            ).scalar()
        assert count == 1


class TestModeFlagPassedThrough:
    def test_backtest_mode_passed_to_agent_pipeline(self, db_engine) -> None:
        """mode='backtest' is forwarded to run_agent_pipeline."""
        with patch("weekly_run.run_agent_pipeline", side_effect=_make_agent_mock()) as mock_agent:
            run_weekly(_DATE, "backtest", db_engine)
            _, kwargs = mock_agent.call_args
            assert kwargs.get("mode") == "backtest"

    def test_live_mode_passed_to_agent_pipeline(self, db_engine) -> None:
        """mode='live' is forwarded to run_agent_pipeline."""
        with patch("weekly_run.run_agent_pipeline", side_effect=_make_agent_mock()) as mock_agent:
            run_weekly(_DATE, "live", db_engine)
            _, kwargs = mock_agent.call_args
            assert kwargs.get("mode") == "live"

    def test_backtest_mode_passed_to_optimizer(self, db_engine) -> None:
        with patch(
            "weekly_run.run_optimization_pipeline", side_effect=_make_optimizer_mock()
        ) as mock_opt:
            run_weekly(_DATE, "backtest", db_engine)
            _, kwargs = mock_opt.call_args
            assert kwargs.get("mode") == "backtest"

    def test_live_mode_passed_to_optimizer(self, db_engine) -> None:
        with patch(
            "weekly_run.run_optimization_pipeline", side_effect=_make_optimizer_mock()
        ) as mock_opt:
            run_weekly(_DATE, "live", db_engine)
            _, kwargs = mock_opt.call_args
            assert kwargs.get("mode") == "live"


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestAlreadyExecuted:
    def test_returns_false_on_empty_db(self, db_engine) -> None:
        assert _already_executed(_DATE_OBJ, db_engine) is False

    def test_returns_false_with_only_weights(self, db_engine) -> None:
        with Session(db_engine) as session:
            session.add(TargetWeight(date=_DATE_OBJ, sector="XLK", weight=0.1))
            session.commit()
        assert _already_executed(_DATE_OBJ, db_engine) is False

    def test_returns_false_with_only_positions(self, db_engine) -> None:
        with Session(db_engine) as session:
            session.add(
                Position(
                    date=_DATE_OBJ,
                    ticker="CASH",
                    shares=1_000_000.0,
                    market_value=0.0,
                    cost_basis=0.0,
                )
            )
            session.commit()
        assert _already_executed(_DATE_OBJ, db_engine) is False

    def test_returns_true_with_both(self, db_engine) -> None:
        with Session(db_engine) as session:
            session.add(TargetWeight(date=_DATE_OBJ, sector="XLK", weight=0.1))
            session.add(
                Position(
                    date=_DATE_OBJ,
                    ticker="CASH",
                    shares=1_000_000.0,
                    market_value=0.0,
                    cost_basis=0.0,
                )
            )
            session.commit()
        assert _already_executed(_DATE_OBJ, db_engine) is True


class TestFetchPricesForDate:
    def test_returns_prices_for_seeded_tickers(self, db_engine) -> None:
        prices = _fetch_prices_for_date(_DATE_OBJ, _TICKERS, db_engine)
        for ticker in _FAKE_PRICES:
            assert prices[ticker] == pytest.approx(_FAKE_PRICES[ticker])

    def test_returns_empty_when_no_prices(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        prices = _fetch_prices_for_date(_DATE_OBJ, _TICKERS, engine)
        assert prices == {}
        engine.dispose()

    def test_uses_most_recent_price_before_date(self, db_engine) -> None:
        """Prices from a day before the rebalance date are used when no same-day row."""
        from db.models import Price

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        prior_date = datetime.date(2024, 6, 6)
        with Session(engine) as session:
            session.add(
                Price(
                    date=prior_date,
                    ticker="XLK",
                    open=195.0,
                    high=195.0,
                    low=195.0,
                    close=195.0,
                    volume=1_000_000,
                    adj_close=195.0,
                )
            )
            session.commit()
        prices = _fetch_prices_for_date(_DATE_OBJ, ["XLK"], engine)
        assert prices["XLK"] == pytest.approx(195.0)
        engine.dispose()


# ---------------------------------------------------------------------------
# Baseline portfolio tests — Ticket 5.2
# ---------------------------------------------------------------------------


class TestNoLlmBaseline:
    """backtest_no_llm: real optimizer with force_zero_views=True; no agent calls."""

    def test_agent_pipeline_not_called(self, db_engine) -> None:
        """No LLM agent calls are made for the no-LLM baseline portfolio."""
        with patch("weekly_run.run_agent_pipeline") as mock_agent:
            run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
            mock_agent.assert_not_called()

    def test_optimizer_called_with_force_zero_views(self, db_engine) -> None:
        """run_optimization_pipeline receives force_zero_views=True."""
        with patch(
            "weekly_run.run_optimization_pipeline", side_effect=_make_optimizer_mock()
        ) as mock_opt:
            run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
            _, kwargs = mock_opt.call_args
            assert kwargs.get("force_zero_views") is True

    def test_llm_cost_is_zero(self, db_engine) -> None:
        result = run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
        assert result["llm_cost_usd"] == pytest.approx(0.0)

    def test_target_weights_written_to_db(self, db_engine) -> None:
        """Optimizer writes TargetWeight rows under the no-LLM portfolio ID."""
        run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(TargetWeight)
                .where(TargetWeight.portfolio_id == PORTFOLIO_BACKTEST_NO_LLM)
                .where(TargetWeight.date == _DATE_OBJ)
            ).scalar()
        assert count == _N

    def test_positions_written_to_correct_portfolio(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == PORTFOLIO_BACKTEST_NO_LLM)
                .where(Position.date == _DATE_OBJ)
            ).scalar()
        assert count > 0

    def test_snapshot_written_to_correct_portfolio(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(PortfolioSnapshot)
                .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_BACKTEST_NO_LLM)
                .where(PortfolioSnapshot.date == _DATE_OBJ)
            ).scalar()
        assert count == 1

    def test_does_not_write_to_live_portfolio(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_NO_LLM)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == "live")
                .where(Position.date == _DATE_OBJ)
            ).scalar()
        assert count == 0


class TestEqualWeightBaseline:
    """backtest_equal_weight: fixed 1/n weights; no agent or BL optimizer calls."""

    @pytest.fixture(autouse=True)
    def _mock_equal_weight_pipeline(self):
        with patch(
            "weekly_run.run_equal_weight_pipeline", side_effect=_make_equal_weight_mock()
        ):
            yield

    def test_agent_pipeline_not_called(self, db_engine) -> None:
        with patch("weekly_run.run_agent_pipeline") as mock_agent:
            run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_EQUAL_WEIGHT)
            mock_agent.assert_not_called()

    def test_optimizer_not_called(self, db_engine) -> None:
        """run_optimization_pipeline is bypassed; equal_weight pipeline is used instead."""
        with patch("weekly_run.run_optimization_pipeline") as mock_opt:
            run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_EQUAL_WEIGHT)
            mock_opt.assert_not_called()

    def test_llm_cost_is_zero(self, db_engine) -> None:
        pid = PORTFOLIO_BACKTEST_EQUAL_WEIGHT
        result = run_weekly(_DATE, "backtest", db_engine, portfolio_id=pid)
        assert result["llm_cost_usd"] == pytest.approx(0.0)

    def test_weights_sum_to_one(self, db_engine) -> None:
        pid = PORTFOLIO_BACKTEST_EQUAL_WEIGHT
        result = run_weekly(_DATE, "backtest", db_engine, portfolio_id=pid)
        assert sum(result["weights_after"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_each_weight_is_one_over_n(self, db_engine) -> None:
        pid = PORTFOLIO_BACKTEST_EQUAL_WEIGHT
        result = run_weekly(_DATE, "backtest", db_engine, portfolio_id=pid)
        expected = 1.0 / _N
        for w in result["weights_after"].values():
            assert w == pytest.approx(expected, abs=1e-9)

    def test_positions_written_to_correct_portfolio(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_EQUAL_WEIGHT)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == PORTFOLIO_BACKTEST_EQUAL_WEIGHT)
                .where(Position.date == _DATE_OBJ)
            ).scalar()
        assert count > 0

    def test_does_not_write_to_live_portfolio(self, db_engine) -> None:
        run_weekly(_DATE, "backtest", db_engine, portfolio_id=PORTFOLIO_BACKTEST_EQUAL_WEIGHT)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == "live")
                .where(Position.date == _DATE_OBJ)
            ).scalar()
        assert count == 0
