"""Tests for src/execution/state.py — Ticket 4.1."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from config import load_config
from db.models import Base, Position
from execution.state import (
    compute_current_weights,
    get_current_positions,
    get_portfolio_value,
    write_portfolio_snapshot,
    write_positions,
)

_UNIVERSE = load_config("universe").ticker_list
_INITIAL_CAPITAL = load_config("backtest").initial_capital


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# Mandatory acceptance-criteria tests
# ---------------------------------------------------------------------------


class TestFirstRunReturnsAllCash:
    def test_first_run_returns_all_cash(self, db: Session) -> None:
        """Empty DB → CASH == initial_capital, all universe tickers == 0."""
        positions = get_current_positions("2024-01-01", db)
        assert positions["CASH"] == pytest.approx(_INITIAL_CAPITAL)
        for ticker in _UNIVERSE:
            assert positions[ticker] == pytest.approx(0.0)

    def test_first_run_any_date_is_consistent(self, db: Session) -> None:
        """First-run behaviour is date-independent — always reads from config."""
        p1 = get_current_positions("2023-06-01", db)
        p2 = get_current_positions("2025-12-31", db)
        assert p1 == p2


class TestWriteThenReadRoundtrip:
    def test_write_then_read_roundtrip(self, db: Session) -> None:
        """Positions written then read back for the exact same date match exactly."""
        positions_in = {"CASH": 50_000.0, "XLK": 100.0, "XLF": 200.0}
        write_positions("2024-01-05", positions_in, db)
        positions_out = get_current_positions("2024-01-05", db)
        assert positions_out == pytest.approx(positions_in)

    def test_read_on_later_date_returns_last_known(self, db: Session) -> None:
        """Query on a date with no row returns the most recent prior row."""
        positions_in = {"CASH": 50_000.0, "XLK": 100.0}
        write_positions("2024-01-05", positions_in, db)
        positions_out = get_current_positions("2024-01-10", db)
        assert positions_out == pytest.approx(positions_in)

    def test_read_before_first_row_returns_initial(self, db: Session) -> None:
        """A query date earlier than any DB row falls back to the first-run default."""
        write_positions("2024-06-01", {"CASH": 50_000.0, "XLK": 100.0}, db)
        positions_out = get_current_positions("2024-01-01", db)
        assert positions_out["CASH"] == pytest.approx(_INITIAL_CAPITAL)


class TestWriteIsIdempotentNoDuplicates:
    def test_write_is_idempotent_no_duplicates(self, db: Session) -> None:
        """Writing the same positions twice keeps the row count constant."""
        positions = {"CASH": 50_000.0, "XLK": 100.0}
        write_positions("2024-01-05", positions, db)
        write_positions("2024-01-05", positions, db)

        count = db.execute(select(func.count()).where(Position.date == "2024-01-05")).scalar()
        assert count == len(positions)

    def test_second_write_updates_shares(self, db: Session) -> None:
        """A second write for the same date overwrites shares rather than inserting."""
        write_positions("2024-01-05", {"CASH": 50_000.0, "XLK": 100.0}, db)
        write_positions("2024-01-05", {"CASH": 60_000.0, "XLK": 150.0}, db)
        positions = get_current_positions("2024-01-05", db)
        assert positions["CASH"] == pytest.approx(60_000.0)
        assert positions["XLK"] == pytest.approx(150.0)


class TestWeightsSumToOneIncludingCash:
    def test_weights_sum_to_one_including_cash(self, db: Session) -> None:
        """Weights over all tickers including CASH sum to 1.0 ± 1e-6."""
        positions = {"CASH": 20_000.0, "XLK": 100.0, "XLF": 50.0}
        write_positions("2024-01-05", positions, db)
        prices = {"XLK": 180.0, "XLF": 40.0}
        weights = compute_current_weights("2024-01-05", db, prices)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_cash_weight_correct_when_equal_split(self, db: Session) -> None:
        """CASH and one ETF at equal dollar value → each has 50% weight."""
        write_positions("2024-01-05", {"CASH": 10_000.0, "XLK": 100.0}, db)
        weights = compute_current_weights("2024-01-05", db, {"XLK": 100.0})
        assert weights["CASH"] == pytest.approx(0.5)
        assert weights["XLK"] == pytest.approx(0.5)

    def test_first_run_weights_are_all_cash(self, db: Session) -> None:
        """On an empty DB CASH weight == 1.0 (all universe ETFs have 0 shares)."""
        weights = compute_current_weights("2024-01-01", db, {"XLK": 180.0})
        assert weights["CASH"] == pytest.approx(1.0)
        for ticker in _UNIVERSE:
            assert weights[ticker] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_portfolio_value
# ---------------------------------------------------------------------------


class TestGetPortfolioValue:
    def test_cash_only_value(self, db: Session) -> None:
        write_positions("2024-01-05", {"CASH": 100_000.0}, db)
        value = get_portfolio_value("2024-01-05", db, {})
        assert value == pytest.approx(100_000.0)

    def test_mixed_portfolio_value(self, db: Session) -> None:
        write_positions("2024-01-05", {"CASH": 50_000.0, "XLK": 100.0}, db)
        value = get_portfolio_value("2024-01-05", db, {"XLK": 200.0})
        # 50_000 cash + 100 × 200 = 70_000
        assert value == pytest.approx(70_000.0)

    def test_missing_price_defaults_to_one(self, db: Session) -> None:
        """Tickers absent from prices are valued at 1.0 per share."""
        write_positions("2024-01-05", {"CASH": 0.0, "XLK": 500.0}, db)
        value = get_portfolio_value("2024-01-05", db, {})
        assert value == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# write_portfolio_snapshot
# ---------------------------------------------------------------------------


class TestWritePortfolioSnapshot:
    def test_snapshot_values_correct(self, db: Session) -> None:
        write_positions("2024-01-05", {"CASH": 50_000.0, "XLK": 100.0, "XLF": 200.0}, db)
        prices = {"XLK": 100.0, "XLF": 50.0}
        snap = write_portfolio_snapshot("2024-01-05", db, prices)
        # total = 50_000 + 10_000 + 10_000 = 70_000
        assert snap["total_value"] == pytest.approx(70_000.0)
        assert snap["cash"] == pytest.approx(50_000.0)
        assert snap["gross_exposure"] == pytest.approx(20_000.0)
        assert snap["net_exposure"] == pytest.approx(20_000.0)

    def test_snapshot_is_idempotent(self, db: Session) -> None:
        """Writing a snapshot twice for the same date keeps exactly one row."""
        from db.models import PortfolioSnapshot

        write_positions("2024-01-05", {"CASH": 50_000.0, "XLK": 100.0}, db)
        prices = {"XLK": 100.0}
        write_portfolio_snapshot("2024-01-05", db, prices)
        write_portfolio_snapshot("2024-01-05", db, prices)

        count = db.execute(
            select(func.count())
            .select_from(PortfolioSnapshot)
            .where(PortfolioSnapshot.date == "2024-01-05")
        ).scalar()
        assert count == 1

    def test_snapshot_returns_dict(self, db: Session) -> None:
        write_positions("2024-01-05", {"CASH": 100_000.0}, db)
        snap = write_portfolio_snapshot("2024-01-05", db, {})
        assert set(snap.keys()) == {"total_value", "cash", "gross_exposure", "net_exposure"}
