"""Portfolio isolation tests — Ticket 5.1.

Verifies that all 4 backtest portfolio IDs can write to the same tables without
row collisions, and that existing "live" rows are not touched by backtest writes.

Covers: positions, portfolio_snapshot, target_weights, views, signals,
        trades, risk_events.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from db.models import (
    ALL_BACKTEST_PORTFOLIO_IDS,
    PORTFOLIO_BACKTEST_EQUAL_WEIGHT,
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PORTFOLIO_BACKTEST_SPY,
    PORTFOLIO_LIVE,
    Base,
    PortfolioSnapshot,
    Position,
    RiskEvent,
    Signal,
    TargetWeight,
    Trade,
    View,
)
from execution.state import (
    get_current_positions,
    write_portfolio_snapshot,
    write_positions,
)

_DATE = datetime.date(2025, 6, 13)
_DATE_STR = "2025-06-13"
_PRICES = {"XLK": 200.0, "CASH": 1.0}


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Portfolio ID constants
# ---------------------------------------------------------------------------


class TestPortfolioIdConstants:
    def test_four_backtest_ids_defined(self) -> None:
        assert len(ALL_BACKTEST_PORTFOLIO_IDS) == 4

    def test_backtest_ids_distinct_from_live(self) -> None:
        for pid in ALL_BACKTEST_PORTFOLIO_IDS:
            assert pid != PORTFOLIO_LIVE

    def test_backtest_ids_are_unique(self) -> None:
        assert len(set(ALL_BACKTEST_PORTFOLIO_IDS)) == 4

    def test_expected_ids_present(self) -> None:
        assert PORTFOLIO_BACKTEST_FULL in ALL_BACKTEST_PORTFOLIO_IDS
        assert PORTFOLIO_BACKTEST_NO_LLM in ALL_BACKTEST_PORTFOLIO_IDS
        assert PORTFOLIO_BACKTEST_EQUAL_WEIGHT in ALL_BACKTEST_PORTFOLIO_IDS
        assert PORTFOLIO_BACKTEST_SPY in ALL_BACKTEST_PORTFOLIO_IDS


# ---------------------------------------------------------------------------
# Positions isolation
# ---------------------------------------------------------------------------


class TestPositionsIsolation:
    def test_four_portfolios_write_same_date_no_collision(self, db) -> None:
        """All 4 backtest IDs + live write positions on the same date without clash."""
        all_ids = ALL_BACKTEST_PORTFOLIO_IDS + [PORTFOLIO_LIVE]
        for pid in all_ids:
            with Session(db) as session:
                write_positions(
                    _DATE_STR, {"CASH": float(len(pid)), "XLK": 5.0}, session, portfolio_id=pid
                )

        with Session(db) as session:
            total = session.execute(select(func.count()).select_from(Position)).scalar()
        # 5 portfolios × 2 tickers each = 10 rows
        assert total == 10

    def test_reads_are_isolated_per_portfolio(self, db) -> None:
        """get_current_positions returns only the requested portfolio's data."""
        with Session(db) as session:
            write_positions(_DATE_STR, {"CASH": 111.0}, session, portfolio_id=PORTFOLIO_LIVE)
            write_positions(
                _DATE_STR, {"CASH": 999.0}, session, portfolio_id=PORTFOLIO_BACKTEST_FULL
            )

        with Session(db) as session:
            live_pos = get_current_positions(_DATE_STR, session, portfolio_id=PORTFOLIO_LIVE)
            bt_pos = get_current_positions(_DATE_STR, session, portfolio_id=PORTFOLIO_BACKTEST_FULL)

        assert live_pos["CASH"] == pytest.approx(111.0)
        assert bt_pos["CASH"] == pytest.approx(999.0)

    def test_backtest_write_does_not_touch_live_rows(self, db) -> None:
        """Writing backtest positions leaves live positions unchanged."""
        with Session(db) as session:
            write_positions(_DATE_STR, {"CASH": 1_000_000.0}, session, portfolio_id=PORTFOLIO_LIVE)

        with Session(db) as session:
            write_positions(
                _DATE_STR, {"CASH": 42.0}, session, portfolio_id=PORTFOLIO_BACKTEST_FULL
            )

        with Session(db) as session:
            live_pos = get_current_positions(_DATE_STR, session, portfolio_id=PORTFOLIO_LIVE)

        assert live_pos["CASH"] == pytest.approx(1_000_000.0)

    def test_default_portfolio_id_is_live(self, db) -> None:
        """Calling write_positions without portfolio_id writes to 'live'."""
        with Session(db) as session:
            write_positions(_DATE_STR, {"CASH": 500.0}, session)  # no portfolio_id

        with Session(db) as session:
            rows = (
                session.execute(select(Position).where(Position.portfolio_id == PORTFOLIO_LIVE))
                .scalars()
                .all()
            )
        assert len(rows) == 1
        assert rows[0].shares == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Portfolio snapshot isolation
# ---------------------------------------------------------------------------


class TestSnapshotIsolation:
    def test_all_backtest_ids_write_same_date(self, db) -> None:
        for i, pid in enumerate(ALL_BACKTEST_PORTFOLIO_IDS):
            with Session(db) as session:
                write_positions(
                    _DATE_STR, {"CASH": float(i + 1) * 1000, "XLK": 0.0}, session, portfolio_id=pid
                )
                write_portfolio_snapshot(
                    _DATE_STR, session, {"XLK": 200.0, "CASH": 1.0}, portfolio_id=pid
                )

        with Session(db) as session:
            total = session.execute(select(func.count()).select_from(PortfolioSnapshot)).scalar()
        assert total == 4

    def test_live_snapshot_unchanged_by_backtest_write(self, db) -> None:
        with Session(db) as session:
            write_positions(
                _DATE_STR, {"CASH": 1_000_000.0, "XLK": 0.0}, session, portfolio_id=PORTFOLIO_LIVE
            )
            write_portfolio_snapshot(
                _DATE_STR, session, {"XLK": 200.0, "CASH": 1.0}, portfolio_id=PORTFOLIO_LIVE
            )

        with Session(db) as session:
            write_positions(
                _DATE_STR, {"CASH": 50.0, "XLK": 0.0}, session, portfolio_id=PORTFOLIO_BACKTEST_FULL
            )
            write_portfolio_snapshot(
                _DATE_STR,
                session,
                {"XLK": 200.0, "CASH": 1.0},
                portfolio_id=PORTFOLIO_BACKTEST_FULL,
            )

        with Session(db) as session:
            live_snap = session.execute(
                select(PortfolioSnapshot.total_value)
                .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_LIVE)
                .where(PortfolioSnapshot.date == _DATE)
            ).scalar()
        assert live_snap == pytest.approx(1_000_000.0)


# ---------------------------------------------------------------------------
# TargetWeight isolation
# ---------------------------------------------------------------------------


class TestTargetWeightIsolation:
    def test_all_portfolios_write_same_date_same_sector(self, db) -> None:
        all_ids = ALL_BACKTEST_PORTFOLIO_IDS + [PORTFOLIO_LIVE]
        with Session(db) as session:
            for pid in all_ids:
                session.add(TargetWeight(portfolio_id=pid, date=_DATE, sector="XLK", weight=0.1))
            session.commit()

        with Session(db) as session:
            total = session.execute(select(func.count()).select_from(TargetWeight)).scalar()
        assert total == 5

    def test_weight_values_isolated_per_portfolio(self, db) -> None:
        with Session(db) as session:
            session.add(
                TargetWeight(portfolio_id=PORTFOLIO_LIVE, date=_DATE, sector="XLK", weight=0.30)
            )
            session.add(
                TargetWeight(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL, date=_DATE, sector="XLK", weight=0.20
                )
            )
            session.commit()

        with Session(db) as session:
            live_w = session.execute(
                select(TargetWeight.weight)
                .where(TargetWeight.portfolio_id == PORTFOLIO_LIVE)
                .where(TargetWeight.date == _DATE)
                .where(TargetWeight.sector == "XLK")
            ).scalar()
            bt_w = session.execute(
                select(TargetWeight.weight)
                .where(TargetWeight.portfolio_id == PORTFOLIO_BACKTEST_FULL)
                .where(TargetWeight.date == _DATE)
                .where(TargetWeight.sector == "XLK")
            ).scalar()

        assert live_w == pytest.approx(0.30)
        assert bt_w == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# View isolation
# ---------------------------------------------------------------------------


class TestViewIsolation:
    def test_all_portfolios_write_same_date_same_sector(self, db) -> None:
        all_ids = ALL_BACKTEST_PORTFOLIO_IDS + [PORTFOLIO_LIVE]
        with Session(db) as session:
            for pid in all_ids:
                session.add(
                    View(
                        portfolio_id=pid,
                        date=_DATE,
                        sector="XLK",
                        expected_return=0.05,
                        confidence=0.7,
                    )
                )
            session.commit()

        with Session(db) as session:
            total = session.execute(select(func.count()).select_from(View)).scalar()
        assert total == 5


# ---------------------------------------------------------------------------
# Signal isolation
# ---------------------------------------------------------------------------


class TestSignalIsolation:
    def test_signals_isolated_per_portfolio(self, db) -> None:
        with Session(db) as session:
            session.add(
                Signal(
                    portfolio_id=PORTFOLIO_LIVE,
                    date=_DATE,
                    agent_name="sentiment",
                    target="XLK",
                    signal_value=0.5,
                    confidence=0.8,
                )
            )
            session.add(
                Signal(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL,
                    date=_DATE,
                    agent_name="sentiment",
                    target="XLK",
                    signal_value=0.0,
                    confidence=0.0,
                )
            )
            session.commit()

        with Session(db) as session:
            live_sig = session.execute(
                select(Signal.signal_value)
                .where(Signal.portfolio_id == PORTFOLIO_LIVE)
                .where(Signal.date == _DATE)
                .where(Signal.agent_name == "sentiment")
                .where(Signal.target == "XLK")
            ).scalar()
            bt_sig = session.execute(
                select(Signal.signal_value)
                .where(Signal.portfolio_id == PORTFOLIO_BACKTEST_FULL)
                .where(Signal.date == _DATE)
                .where(Signal.agent_name == "sentiment")
                .where(Signal.target == "XLK")
            ).scalar()

        assert live_sig == pytest.approx(0.5)
        assert bt_sig == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trade isolation
# ---------------------------------------------------------------------------


class TestTradeIsolation:
    def test_trades_isolated_per_portfolio(self, db) -> None:
        with Session(db) as session:
            session.add(
                Trade(
                    portfolio_id=PORTFOLIO_LIVE,
                    date=_DATE,
                    ticker="XLK",
                    side="buy",
                    shares=50,
                    price=200.0,
                    commission=1.0,
                    slippage=0.0,
                )
            )
            session.add(
                Trade(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL,
                    date=_DATE,
                    ticker="XLK",
                    side="buy",
                    shares=100,
                    price=200.0,
                    commission=2.0,
                    slippage=0.0,
                )
            )
            session.commit()

        with Session(db) as session:
            live_count = session.execute(
                select(func.count()).select_from(Trade).where(Trade.portfolio_id == PORTFOLIO_LIVE)
            ).scalar()
            bt_count = session.execute(
                select(func.count())
                .select_from(Trade)
                .where(Trade.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            ).scalar()
            total = session.execute(select(func.count()).select_from(Trade)).scalar()

        assert live_count == 1
        assert bt_count == 1
        assert total == 2


# ---------------------------------------------------------------------------
# RiskEvent isolation
# ---------------------------------------------------------------------------


class TestRiskEventIsolation:
    def test_risk_events_isolated_per_portfolio(self, db) -> None:
        with Session(db) as session:
            session.add(
                RiskEvent(
                    portfolio_id=PORTFOLIO_LIVE,
                    date=_DATE,
                    check_name="max_position",
                    triggered=False,
                    value=0.1,
                    threshold=0.25,
                    action_taken="none",
                    message="ok",
                )
            )
            session.add(
                RiskEvent(
                    portfolio_id=PORTFOLIO_BACKTEST_FULL,
                    date=_DATE,
                    check_name="max_position",
                    triggered=True,
                    value=0.30,
                    threshold=0.25,
                    action_taken="clip_and_renorm",
                    message="triggered",
                )
            )
            session.commit()

        with Session(db) as session:
            live_triggered = session.execute(
                select(RiskEvent.triggered)
                .where(RiskEvent.portfolio_id == PORTFOLIO_LIVE)
                .where(RiskEvent.date == _DATE)
            ).scalar()
            bt_triggered = session.execute(
                select(RiskEvent.triggered)
                .where(RiskEvent.portfolio_id == PORTFOLIO_BACKTEST_FULL)
                .where(RiskEvent.date == _DATE)
            ).scalar()

        assert live_triggered is False
        assert bt_triggered is True


# ---------------------------------------------------------------------------
# Cross-portfolio non-overlap: querying one ID returns only its rows
# ---------------------------------------------------------------------------


class TestCrossPortfolioNonOverlap:
    def test_all_four_backtest_ids_are_non_overlapping(self, db) -> None:
        """Each backtest portfolio writes distinct position rows; reads never bleed."""
        amounts = {
            PORTFOLIO_BACKTEST_FULL: 1_000_000.0,
            PORTFOLIO_BACKTEST_NO_LLM: 2_000_000.0,
            PORTFOLIO_BACKTEST_EQUAL_WEIGHT: 3_000_000.0,
            PORTFOLIO_BACKTEST_SPY: 4_000_000.0,
        }
        for pid, cash in amounts.items():
            with Session(db) as session:
                write_positions(_DATE_STR, {"CASH": cash}, session, portfolio_id=pid)

        for pid, expected_cash in amounts.items():
            with Session(db) as session:
                pos = get_current_positions(_DATE_STR, session, portfolio_id=pid)
            assert pos["CASH"] == pytest.approx(expected_cash), (
                f"portfolio_id={pid} returned wrong CASH: expected {expected_cash}, got {pos['CASH']}"  # noqa: E501
            )

    def test_live_data_invisible_to_backtest_queries(self, db) -> None:
        """Live rows written before backtest runs are invisible to backtest reads."""
        with Session(db) as session:
            write_positions(
                _DATE_STR, {"CASH": 999_999.0, "XLK": 10.0}, session, portfolio_id=PORTFOLIO_LIVE
            )

        with Session(db) as session:
            bt_pos = get_current_positions(_DATE_STR, session, portfolio_id=PORTFOLIO_BACKTEST_FULL)

        # backtest_full has no positions → should get initial_capital fallback, not live data
        from config import load_config

        initial_capital = load_config("backtest").initial_capital
        assert bt_pos["CASH"] == pytest.approx(initial_capital)
        # The live XLK=10 must NOT appear in backtest reads
        assert bt_pos.get("XLK", 0.0) == pytest.approx(0.0)
