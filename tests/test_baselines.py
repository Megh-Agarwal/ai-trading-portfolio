"""Tests for src/baselines.py — Ticket 5.3.

All tests use an in-memory SQLite DB with seeded SPY prices.  No real
ingestion or API calls occur.  Expected values are computed from the same
formulas used in the implementation so the tests survive config changes.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from baselines import _fetch_spy_close, run_spy_benchmark
from config import load_config
from db.models import (
    PORTFOLIO_BACKTEST_SPY,
    PORTFOLIO_LIVE,
    Base,
    PortfolioSnapshot,
    Position,
    Price,
    Trade,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATE_W1 = datetime.date(2025, 6, 13)
_DATE_W2 = datetime.date(2025, 6, 20)
_DATE_W1_STR = "2025-06-13"
_DATE_W2_STR = "2025-06-20"

_SPY_PRICE_W1 = 400.0
_SPY_PRICE_W2 = 440.0  # +10% — a clean, easy-to-reason-about move


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _seed_spy_price(engine, date: datetime.date, price: float) -> None:
    with Session(engine) as session:
        session.merge(
            Price(
                date=date,
                ticker="SPY",
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10_000_000,
                adj_close=price,
            )
        )
        session.commit()


def _get_spy_position(engine) -> float:
    """Return the SPY share count from the most recent position row."""
    with Session(engine) as session:
        row = session.execute(
            select(Position.shares)
            .where(Position.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            .where(Position.ticker == "SPY")
            .order_by(Position.date.desc())
            .limit(1)
        ).scalar()
    return float(row) if row is not None else 0.0


def _get_cash_position(engine) -> float:
    with Session(engine) as session:
        row = session.execute(
            select(Position.shares)
            .where(Position.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            .where(Position.ticker == "CASH")
            .order_by(Position.date.desc())
            .limit(1)
        ).scalar()
    return float(row) if row is not None else 0.0


# ---------------------------------------------------------------------------
# TestFetchSpyClose
# ---------------------------------------------------------------------------


class TestFetchSpyClose:
    def test_returns_seeded_price(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        price = _fetch_spy_close(_DATE_W1, db_engine)
        assert price == pytest.approx(_SPY_PRICE_W1)

    def test_returns_none_when_no_price(self, db_engine) -> None:
        price = _fetch_spy_close(_DATE_W1, db_engine)
        assert price is None

    def test_returns_most_recent_before_date(self, db_engine) -> None:
        """Returns week-1 price when querying a date between the two seeded rows."""
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        _seed_spy_price(db_engine, _DATE_W2, _SPY_PRICE_W2)
        between = datetime.date(2025, 6, 17)  # between W1 and W2
        price = _fetch_spy_close(between, db_engine)
        assert price == pytest.approx(_SPY_PRICE_W1)


# ---------------------------------------------------------------------------
# TestFirstRun
# ---------------------------------------------------------------------------


class TestFirstRun:
    def test_spy_position_written(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        spy_shares = _get_spy_position(db_engine)
        assert spy_shares > 0

    def test_one_trade_written(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Trade)
                .where(Trade.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            ).scalar()
        assert count == 1

    def test_trade_is_a_buy(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        with Session(db_engine) as session:
            trade = session.execute(
                select(Trade).where(Trade.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            ).scalar_one()
        assert trade.side == "buy"
        assert trade.ticker == "SPY"

    def test_shares_computed_from_initial_capital_and_price(self, db_engine) -> None:
        """shares = floor(initial_capital / (price × (1 + cost_rate)))."""
        cfg_bt = load_config("backtest")
        cfg_opt = load_config("optimizer")
        tc = cfg_opt.transaction_costs
        cost_rate = (tc.spread_bps + tc.slippage_bps) / 10_000.0
        import math

        expected_shares = math.floor(
            float(cfg_bt.initial_capital) / (_SPY_PRICE_W1 * (1.0 + cost_rate))
        )
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        assert _get_spy_position(db_engine) == pytest.approx(expected_shares)

    def test_cash_is_nonnegative(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        assert _get_cash_position(db_engine) >= 0.0

    def test_raises_without_spy_price(self, db_engine) -> None:
        with pytest.raises(RuntimeError, match="No SPY price"):
            run_spy_benchmark(_DATE_W1_STR, db_engine)

    def test_snapshot_written(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(PortfolioSnapshot)
                .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_BACKTEST_SPY)
                .where(PortfolioSnapshot.date == _DATE_W1)
            ).scalar()
        assert count == 1


# ---------------------------------------------------------------------------
# TestHoldWeeks — acceptance criteria
# ---------------------------------------------------------------------------


class TestHoldWeeks:
    def test_no_new_trade_on_subsequent_run(self, db_engine) -> None:
        """No trades written after week 1 — hold unchanged."""
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        _seed_spy_price(db_engine, _DATE_W2, _SPY_PRICE_W2)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        run_spy_benchmark(_DATE_W2_STR, db_engine)

        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Trade)
                .where(Trade.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            ).scalar()
        assert count == 1  # only the week-1 buy

    def test_spy_shares_unchanged_after_week2(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        _seed_spy_price(db_engine, _DATE_W2, _SPY_PRICE_W2)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        shares_w1 = _get_spy_position(db_engine)

        run_spy_benchmark(_DATE_W2_STR, db_engine)
        shares_w2 = _get_spy_position(db_engine)

        assert shares_w2 == pytest.approx(shares_w1)

    def test_portfolio_value_tracks_spy_price_exactly(self, db_engine) -> None:
        """Week-2 total_value = spy_shares × spy_price_w2 + cash (no approximation)."""
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        _seed_spy_price(db_engine, _DATE_W2, _SPY_PRICE_W2)
        run_spy_benchmark(_DATE_W1_STR, db_engine)

        spy_shares = _get_spy_position(db_engine)
        cash = _get_cash_position(db_engine)

        result_w2 = run_spy_benchmark(_DATE_W2_STR, db_engine)

        expected_w2 = spy_shares * _SPY_PRICE_W2 + cash
        assert result_w2["ending_portfolio_value"] == pytest.approx(expected_w2)

    def test_value_change_equals_shares_times_price_delta(self, db_engine) -> None:
        """Key acceptance criterion: Δvalue = shares × (price_w2 − price_w1).

        Cash does not change between weeks (no dividends, no rebalancing),
        so the entire portfolio value change comes from the price move on the
        fixed SPY position.
        """
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        _seed_spy_price(db_engine, _DATE_W2, _SPY_PRICE_W2)

        result_w1 = run_spy_benchmark(_DATE_W1_STR, db_engine)
        spy_shares = _get_spy_position(db_engine)

        result_w2 = run_spy_benchmark(_DATE_W2_STR, db_engine)

        expected_delta = spy_shares * (_SPY_PRICE_W2 - _SPY_PRICE_W1)
        actual_delta = result_w2["ending_portfolio_value"] - result_w1["ending_portfolio_value"]
        assert actual_delta == pytest.approx(expected_delta)

    def test_snapshot_written_for_each_week(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        _seed_spy_price(db_engine, _DATE_W2, _SPY_PRICE_W2)
        run_spy_benchmark(_DATE_W1_STR, db_engine)
        run_spy_benchmark(_DATE_W2_STR, db_engine)

        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(PortfolioSnapshot)
                .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            ).scalar()
        assert count == 2


# ---------------------------------------------------------------------------
# TestIsolation
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_spy_does_not_write_to_live_portfolio(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)

        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == PORTFOLIO_LIVE)
            ).scalar()
        assert count == 0

    def test_portfolio_id_is_set_correctly(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        run_spy_benchmark(_DATE_W1_STR, db_engine)

        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Position)
                .where(Position.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            ).scalar()
        assert count > 0

    def test_idempotent_second_call_same_date(self, db_engine) -> None:
        """Re-running for the same date produces identical values, no duplicate rows."""
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        r1 = run_spy_benchmark(_DATE_W1_STR, db_engine)
        r2 = run_spy_benchmark(_DATE_W1_STR, db_engine)

        assert r1["ending_portfolio_value"] == pytest.approx(r2["ending_portfolio_value"])
        # Idempotent: still only one trade
        with Session(db_engine) as session:
            count = session.execute(
                select(func.count())
                .select_from(Trade)
                .where(Trade.portfolio_id == PORTFOLIO_BACKTEST_SPY)
            ).scalar()
        assert count == 1

    def test_result_dict_has_required_keys(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        result = run_spy_benchmark(_DATE_W1_STR, db_engine)
        required_keys = ("date", "mode", "skipped", "llm_cost_usd", "weights_after", "ending_portfolio_value")  # noqa: E501
        for key in required_keys:
            assert key in result, f"missing key: {key}"

    def test_llm_cost_always_zero(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        result = run_spy_benchmark(_DATE_W1_STR, db_engine)
        assert result["llm_cost_usd"] == pytest.approx(0.0)

    def test_weights_after_is_spy_one(self, db_engine) -> None:
        _seed_spy_price(db_engine, _DATE_W1, _SPY_PRICE_W1)
        result = run_spy_benchmark(_DATE_W1_STR, db_engine)
        assert result["weights_after"] == {"SPY": 1.0}
