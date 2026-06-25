"""Tests for src/backtest/engine.py — Ticket 5.4.

All DB tests use in-memory SQLite.  run_weekly is monkeypatched in dispatch
and resumability tests to avoid running the full agent/optimizer stack.
Look-ahead tests hit real DB helpers to verify date-scoped queries.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backtest.engine import _generate_fridays, run_backtest
from db.models import (
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_BACKTEST_SPY,
    Base,
    Position,
    TargetWeight,
)
from weekly_run import _already_executed, _fetch_prices_for_date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W1 = datetime.date(2025, 6, 13)  # a Friday
_W2 = datetime.date(2025, 6, 20)  # next Friday
_W3 = datetime.date(2025, 6, 27)  # third Friday
_W1_STR = "2025-06-13"
_W2_STR = "2025-06-20"

_ALL_PORTFOLIOS = [
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_SPY,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
    return {
        "skipped": False,
        "ending_portfolio_value": 1_000_000.0,
        "llm_cost_usd": 0.01 if portfolio_id == PORTFOLIO_BACKTEST_FULL else 0.0,
    }


def _seed_price(engine, date: datetime.date, ticker: str, price: float) -> None:
    from db.models import Price

    with Session(engine) as session:
        session.merge(
            Price(
                date=date,
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


def _seed_completed_week(engine, date: datetime.date, portfolio_id: str) -> None:
    """Write the minimum rows for _already_executed to return True."""
    with Session(engine) as session:
        session.merge(
            TargetWeight(portfolio_id=portfolio_id, date=date, sector="XLK", weight=0.1)
        )
        session.merge(
            Position(
                portfolio_id=portfolio_id,
                date=date,
                ticker="XLK",
                shares=100.0,
                market_value=1_000.0,
                cost_basis=950.0,
            )
        )
        session.commit()


# ---------------------------------------------------------------------------
# TestGenerateFridays
# ---------------------------------------------------------------------------


class TestGenerateFridays:
    def test_single_friday(self) -> None:
        assert _generate_fridays(_W1, _W1) == [_W1]

    def test_two_fridays(self) -> None:
        result = _generate_fridays(_W1, _W2)
        assert result == [_W1, _W2]

    def test_three_fridays(self) -> None:
        result = _generate_fridays(_W1, _W3)
        assert result == [_W1, _W2, _W3]

    def test_start_on_friday_included(self) -> None:
        assert _generate_fridays(_W1, _W1)[0] == _W1

    def test_start_not_on_friday_advances_to_next(self) -> None:
        monday = datetime.date(2025, 6, 9)  # Monday before _W1
        result = _generate_fridays(monday, _W1)
        assert result == [_W1]

    def test_end_before_any_friday_returns_empty(self) -> None:
        # Monday through Thursday — no Friday in range
        monday = datetime.date(2025, 6, 9)
        thursday = datetime.date(2025, 6, 12)
        assert _generate_fridays(monday, thursday) == []

    def test_all_returned_dates_are_fridays(self) -> None:
        start = datetime.date(2024, 7, 5)
        end = datetime.date(2026, 6, 12)
        fridays = _generate_fridays(start, end)
        for d in fridays:
            assert d.weekday() == 4, f"{d} is not a Friday"

    def test_102_fridays_in_full_backtest_window(self) -> None:
        start = datetime.date(2024, 7, 5)
        end = datetime.date(2026, 6, 12)
        fridays = _generate_fridays(start, end)
        assert len(fridays) == 102


# ---------------------------------------------------------------------------
# TestRunBacktest — dispatch and return contract
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_calls_run_weekly_for_each_portfolio_and_date(
        self, db_engine, monkeypatch
    ) -> None:
        calls: list[tuple[str, str]] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            calls.append((date_str, portfolio_id))
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        run_backtest(_W1, _W1, db_engine)

        # One call per portfolio ID for the single Friday
        assert len(calls) == 4
        called_portfolios = {pid for _, pid in calls}
        assert called_portfolios == set(_ALL_PORTFOLIOS)

    def test_full_portfolio_runs_first_per_date(self, db_engine, monkeypatch) -> None:
        """PORTFOLIO_BACKTEST_FULL must be dispatched before the other portfolios
        so its LLM cache is warm when the no-LLM/equal-weight/SPY runs execute."""
        order: list[str] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            order.append(portfolio_id)
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        run_backtest(_W1, _W1, db_engine)

        assert order[0] == PORTFOLIO_BACKTEST_FULL

    def test_mode_is_always_backtest(self, db_engine, monkeypatch) -> None:
        modes: list[str] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            modes.append(mode)
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        run_backtest(_W1, _W1, db_engine)

        assert all(m == "backtest" for m in modes)

    def test_skip_ingest_is_always_true(self, db_engine, monkeypatch) -> None:
        ingest_flags: list[bool] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            ingest_flags.append(skip_ingest)
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        run_backtest(_W1, _W1, db_engine)

        assert all(flag is True for flag in ingest_flags)

    def test_two_weeks_dispatches_eight_calls(self, db_engine, monkeypatch) -> None:
        calls: list[tuple[str, str]] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            calls.append((date_str, portfolio_id))
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        run_backtest(_W1, _W2, db_engine)

        assert len(calls) == 8  # 2 dates × 4 portfolios

    def test_returns_required_keys(self, db_engine, monkeypatch) -> None:
        monkeypatch.setattr("backtest.engine.run_weekly", _fake_run_weekly)
        result = run_backtest(_W1, _W1, db_engine)

        for key in (
            "start_date",
            "end_date",
            "weeks_total",
            "portfolio_weeks_completed",
            "portfolio_weeks_skipped",
            "total_llm_cost_usd",
        ):
            assert key in result, f"missing key: {key}"

    def test_weeks_total_matches_friday_count(self, db_engine, monkeypatch) -> None:
        monkeypatch.setattr("backtest.engine.run_weekly", _fake_run_weekly)
        result = run_backtest(_W1, _W2, db_engine)
        assert result["weeks_total"] == 2

    def test_accumulates_llm_cost(self, db_engine, monkeypatch) -> None:
        monkeypatch.setattr("backtest.engine.run_weekly", _fake_run_weekly)
        result = run_backtest(_W1, _W2, db_engine)
        # full portfolio contributes 0.01 per week; 2 weeks = 0.02
        assert result["total_llm_cost_usd"] == pytest.approx(0.02)

    def test_completed_count_per_portfolio(self, db_engine, monkeypatch) -> None:
        monkeypatch.setattr("backtest.engine.run_weekly", _fake_run_weekly)
        result = run_backtest(_W1, _W2, db_engine)
        for pid in _ALL_PORTFOLIOS:
            assert result["portfolio_weeks_completed"][pid] == 2

    def test_skipped_count_incremented_when_run_weekly_returns_skipped(
        self, db_engine, monkeypatch
    ) -> None:
        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            if portfolio_id == PORTFOLIO_BACKTEST_FULL:
                return {"skipped": True, "reason": "already done", "date": date_str}
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        result = run_backtest(_W1, _W1, db_engine)

        assert result["portfolio_weeks_skipped"][PORTFOLIO_BACKTEST_FULL] == 1
        assert result["portfolio_weeks_completed"][PORTFOLIO_BACKTEST_FULL] == 0
        assert result["portfolio_weeks_completed"][PORTFOLIO_BACKTEST_NO_LLM] == 1


# ---------------------------------------------------------------------------
# TestNoLookahead — data-layer date scoping
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_future_price_not_visible_on_earlier_date(self, db_engine) -> None:
        """_fetch_prices_for_date must return W1 price, not the W2 future price."""
        _seed_price(db_engine, _W1, "SPY", 400.0)
        _seed_price(db_engine, _W2, "SPY", 500.0)  # future — must be invisible on W1

        prices = _fetch_prices_for_date(_W1, ["SPY"], db_engine)

        assert prices["SPY"] == pytest.approx(400.0)

    def test_future_position_not_visible_on_earlier_date(self, db_engine) -> None:
        """get_current_positions queried on W1 must not see a W2 position."""
        from execution.state import get_current_positions

        # Write W1 position (50 shares) and a future W2 position (100 shares)
        with Session(db_engine) as session:
            session.merge(
                Position(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL,
                    date=_W1,
                    ticker="XLK",
                    shares=50.0,
                    market_value=5_000.0,
                    cost_basis=4_800.0,
                )
            )
            session.merge(
                Position(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL,
                    date=_W2,
                    ticker="XLK",
                    shares=100.0,
                    market_value=10_000.0,
                    cost_basis=9_600.0,
                )
            )
            session.commit()

        with Session(db_engine) as session:
            positions = get_current_positions(
                _W1_STR, session, portfolio_id=PORTFOLIO_BACKTEST_FULL
            )

        assert positions.get("XLK", 0.0) == pytest.approx(50.0)  # W1 value, not W2

    def test_only_week1_price_used_when_querying_week1(self, db_engine) -> None:
        """Multiple tickers: each should return its W1 price even when W2 prices exist."""
        for ticker, price_w1, price_w2 in [("XLK", 200.0, 220.0), ("XLV", 150.0, 160.0)]:
            _seed_price(db_engine, _W1, ticker, price_w1)
            _seed_price(db_engine, _W2, ticker, price_w2)

        prices = _fetch_prices_for_date(_W1, ["XLK", "XLV"], db_engine)

        assert prices["XLK"] == pytest.approx(200.0)
        assert prices["XLV"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# TestResumability
# ---------------------------------------------------------------------------


class TestResumability:
    def test_pre_populated_week_is_skipped(self, db_engine, monkeypatch) -> None:
        """If a (date, portfolio_id) is already in the DB, run_backtest skips it."""
        # Pre-populate W1 for PORTFOLIO_BACKTEST_FULL
        _seed_completed_week(db_engine, _W1, PORTFOLIO_BACKTEST_FULL)

        dispatched: list[tuple[str, str]] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            if _already_executed(
                datetime.date.fromisoformat(date_str), db_engine, portfolio_id=portfolio_id
            ):
                return {"skipped": True, "reason": "already done", "date": date_str}
            dispatched.append((date_str, portfolio_id))
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        result = run_backtest(_W1, _W1, db_engine)

        assert result["portfolio_weeks_skipped"][PORTFOLIO_BACKTEST_FULL] == 1
        assert (str(_W1), PORTFOLIO_BACKTEST_FULL) not in dispatched

    def test_partial_run_completes_remaining_weeks(self, db_engine, monkeypatch) -> None:
        """W1 pre-populated; running W1–W2 range should execute only W2."""
        for pid in _ALL_PORTFOLIOS:
            _seed_completed_week(db_engine, _W1, pid)

        dispatched: list[tuple[str, str]] = []

        def mock_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            if _already_executed(
                datetime.date.fromisoformat(date_str), db_engine, portfolio_id=portfolio_id
            ):
                return {"skipped": True, "reason": "already done", "date": date_str}
            dispatched.append((date_str, portfolio_id))
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", mock_run_weekly)
        result = run_backtest(_W1, _W2, db_engine)

        # All 4 portfolios skipped for W1
        for pid in _ALL_PORTFOLIOS:
            assert result["portfolio_weeks_skipped"][pid] == 1

        # All 4 portfolios executed for W2
        for pid in _ALL_PORTFOLIOS:
            assert result["portfolio_weeks_completed"][pid] == 1

        # Only W2 dispatched
        dispatched_dates = {date_str for date_str, _ in dispatched}
        assert dispatched_dates == {_W2_STR}

    def test_idempotent_second_call_all_skipped(self, db_engine, monkeypatch) -> None:
        """Running the same range twice: second call should skip every (date, portfolio)."""
        call_count: dict[str, int] = {"n": 0}

        def _mock_first(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            call_count["n"] += 1
            _seed_completed_week(db_engine, datetime.date.fromisoformat(date_str), portfolio_id)
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", _mock_first)
        run_backtest(_W1, _W1, db_engine)
        first_call_count = call_count["n"]  # should be 4

        def _mock_second(date_str, mode, db_engine, skip_ingest, portfolio_id, force=False):
            call_count["n"] += 1
            if _already_executed(
                datetime.date.fromisoformat(date_str), db_engine, portfolio_id=portfolio_id
            ):
                return {"skipped": True, "reason": "already done", "date": date_str}
            return _fake_run_weekly(date_str, mode, db_engine, skip_ingest, portfolio_id)

        monkeypatch.setattr("backtest.engine.run_weekly", _mock_second)
        result2 = run_backtest(_W1, _W1, db_engine)

        assert first_call_count == 4  # 1 week × 4 portfolios
        for pid in _ALL_PORTFOLIOS:
            assert result2["portfolio_weeks_skipped"][pid] == 1
