"""Tests for src/eval/attribution.py — Ticket 4.4.

Hand-calculated 2-week scenario
================================
Start  2024-01-01:  XLK=300sh@$100=$30k, XLF=1000sh@$40=$40k, CASH=$30k  → total=$100k
End    2024-01-14:  same shares,  XLK@$110=$33k, XLF@$42=$42k, CASH=$30k → total=$105k

Total return = (105k − 100k) / 100k = 0.05 (5%)

Start weights:  XLK=0.300,  XLF=0.400,  CASH=0.300
End weights:    XLK=33/105=0.31429,  XLF=42/105=0.40,  CASH=30/105=0.28571
Avg weights:    XLK=(0.300+0.31429)/2=0.30714, XLF=0.400, CASH=0.29286

Sector returns: XLK=(110−100)/100=0.10, XLF=(42−40)/40=0.05

Contributions:
  XLK: 0.30714 × 0.10 = 0.030714
  XLF: 0.400   × 0.05 = 0.020000
  CASH:               = 0.000000
  Sum:                = 0.050714

Unexplained (no cost drag): 0.05 − 0.050714 = −0.000714  ← within 0.5% tolerance

One trade mid-period: buy 10 XLK@$105, commission=$6.30
  avg portfolio = (100k + 105k) / 2 = $102,500
  cost_drag_bps = 6.30 / 102_500 × 10_000 ≈ 0.6146 bps
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base, PortfolioSnapshot, Position, Trade
from eval.attribution import (
    compute_cost_drag,
    compute_period_return,
    compute_sector_contribution,
    reconcile_attribution,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


_START = "2024-01-01"
_END = "2024-01-14"

_PRICES = pd.DataFrame(
    {
        "XLK": {_START: 100.0, _END: 110.0},
        "XLF": {_START: 40.0, _END: 42.0},
    }
)


@pytest.fixture()
def scenario(db: Session):
    """Write the hand-calculated scenario into the in-memory DB."""
    start_obj = datetime.date(2024, 1, 1)
    end_obj = datetime.date(2024, 1, 14)

    # Portfolio snapshots
    db.add(PortfolioSnapshot(
        date=start_obj, total_value=100_000.0,
        cash=30_000.0, gross_exposure=70_000.0, net_exposure=70_000.0,
    ))
    db.add(PortfolioSnapshot(
        date=end_obj, total_value=105_000.0,
        cash=30_000.0, gross_exposure=75_000.0, net_exposure=75_000.0,
    ))

    # Positions at start
    for ticker, shares in [("XLK", 300.0), ("XLF", 1000.0), ("CASH", 30_000.0)]:
        db.add(Position(date=start_obj, ticker=ticker,
                        shares=shares, market_value=0.0, cost_basis=0.0))

    # Positions at end (same shares — no rebalance, prices moved)
    for ticker, shares in [("XLK", 300.0), ("XLF", 1000.0), ("CASH", 30_000.0)]:
        db.add(Position(date=end_obj, ticker=ticker,
                        shares=shares, market_value=0.0, cost_basis=0.0))

    # One trade mid-period
    db.add(Trade(
        date=datetime.date(2024, 1, 7), ticker="XLK", side="buy",
        shares=10.0, price=105.0, commission=6.30, slippage=0.0,
    ))

    db.commit()
    return db  # same session, scenario data already committed


# ---------------------------------------------------------------------------
# compute_period_return
# ---------------------------------------------------------------------------


class TestComputePeriodReturn:
    def test_correct_total_return(self, scenario: Session) -> None:
        result = compute_period_return(_START, _END, scenario)
        assert result["total_return_pct"] == pytest.approx(0.05, rel=1e-9)

    def test_start_and_end_values(self, scenario: Session) -> None:
        result = compute_period_return(_START, _END, scenario)
        assert result["start_value"] == pytest.approx(100_000.0)
        assert result["end_value"] == pytest.approx(105_000.0)

    def test_missing_start_snapshot_raises(self, db: Session) -> None:
        with pytest.raises(ValueError, match="start_date"):
            compute_period_return("2023-01-01", _END, db)

    def test_missing_end_snapshot_raises(self, db: Session) -> None:
        db.add(PortfolioSnapshot(
            date=datetime.date(2024, 1, 1), total_value=100_000.0,
            cash=30_000.0, gross_exposure=70_000.0, net_exposure=70_000.0,
        ))
        db.commit()
        with pytest.raises(ValueError, match="end_date"):
            compute_period_return(_START, "2099-12-31", db)

    def test_zero_return_when_values_equal(self, db: Session) -> None:
        for d in [datetime.date(2024, 1, 1), datetime.date(2024, 1, 14)]:
            db.add(PortfolioSnapshot(
                date=d, total_value=100_000.0,
                cash=100_000.0, gross_exposure=0.0, net_exposure=0.0,
            ))
        db.commit()
        result = compute_period_return(_START, _END, db)
        assert result["total_return_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_sector_contribution
# ---------------------------------------------------------------------------


class TestComputeSectorContribution:
    def test_sector_contributions_hand_calculated(self, scenario: Session) -> None:
        contribs = compute_sector_contribution(_START, _END, scenario, _PRICES)
        # XLK: avg_weight ≈ 0.30714, return = 0.10 → contrib ≈ 0.030714
        assert contribs["XLK"] == pytest.approx(0.030714, abs=1e-4)
        # XLF: avg_weight = 0.40, return = 0.05 → contrib = 0.020
        assert contribs["XLF"] == pytest.approx(0.020000, abs=1e-6)
        # CASH always 0
        assert contribs["CASH"] == pytest.approx(0.0)

    def test_contributions_sum_close_to_total_return(self, scenario: Session) -> None:
        """Sum of contributions ≈ total return within 0.5% absolute (approximation)."""
        contribs = compute_sector_contribution(_START, _END, scenario, _PRICES)
        total_return = compute_period_return(_START, _END, scenario)["total_return_pct"]
        assert abs(sum(contribs.values()) - total_return) < 0.005

    def test_cash_contribution_always_zero(self, scenario: Session) -> None:
        contribs = compute_sector_contribution(_START, _END, scenario, _PRICES)
        assert "CASH" in contribs
        assert contribs["CASH"] == pytest.approx(0.0)

    def test_missing_ticker_in_prices_is_skipped(self, scenario: Session) -> None:
        partial_prices = pd.DataFrame({"XLK": {_START: 100.0, _END: 110.0}})
        contribs = compute_sector_contribution(_START, _END, scenario, partial_prices)
        # XLF skipped (not in prices), but XLK and CASH present
        assert "XLK" in contribs
        assert "XLF" not in contribs

    def test_all_cash_portfolio_returns_zero_contributions(self, db: Session) -> None:
        """Portfolio held entirely in CASH contributes 0 from all sectors."""
        for d_obj, total in [
            (datetime.date(2024, 1, 1), 100_000.0),
            (datetime.date(2024, 1, 14), 100_000.0),
        ]:
            db.add(PortfolioSnapshot(
                date=d_obj, total_value=total,
                cash=total, gross_exposure=0.0, net_exposure=0.0,
            ))
            db.add(Position(date=d_obj, ticker="CASH", shares=total,
                            market_value=0.0, cost_basis=0.0))
        db.commit()
        contribs = compute_sector_contribution(_START, _END, db, _PRICES)
        # All sectors: shares=0 → weight=0 → contribution=0
        for ticker, val in contribs.items():
            assert val == pytest.approx(0.0), f"{ticker} should be 0"


# ---------------------------------------------------------------------------
# compute_cost_drag
# ---------------------------------------------------------------------------


class TestComputeCostDrag:
    def test_cost_drag_matches_commission_sum(self, scenario: Session) -> None:
        result = compute_cost_drag(_START, _END, scenario)
        assert result["total_cost_usd"] == pytest.approx(6.30)

    def test_cost_drag_bps_computed_correctly(self, scenario: Session) -> None:
        # avg_portfolio = (100k + 105k) / 2 = 102_500
        # cost_drag_bps = 6.30 / 102_500 × 10_000 ≈ 0.6146
        result = compute_cost_drag(_START, _END, scenario)
        expected_bps = 6.30 / 102_500 * 10_000
        assert result["cost_drag_bps"] == pytest.approx(expected_bps, rel=1e-6)

    def test_no_trades_gives_zero_cost_drag(self, db: Session) -> None:
        for d_obj, total in [
            (datetime.date(2024, 1, 1), 100_000.0),
            (datetime.date(2024, 1, 14), 105_000.0),
        ]:
            db.add(PortfolioSnapshot(
                date=d_obj, total_value=total,
                cash=total, gross_exposure=0.0, net_exposure=0.0,
            ))
        db.commit()
        result = compute_cost_drag(_START, _END, db)
        assert result["total_cost_usd"] == pytest.approx(0.0)
        assert result["cost_drag_bps"] == pytest.approx(0.0)

    def test_excludes_trades_outside_period(self, scenario: Session) -> None:
        # Trade before period
        scenario.add(Trade(
            date=datetime.date(2023, 12, 31), ticker="XLF", side="sell",
            shares=50.0, price=40.0, commission=100.0, slippage=0.0,
        ))
        scenario.commit()
        result = compute_cost_drag(_START, _END, scenario)
        # Only the 6.30 from the fixture trade should be counted
        assert result["total_cost_usd"] == pytest.approx(6.30)

    def test_multiple_trades_summed_correctly(self, scenario: Session) -> None:
        scenario.add(Trade(
            date=datetime.date(2024, 1, 10), ticker="XLF", side="buy",
            shares=25.0, price=41.0, commission=3.075, slippage=0.0,
        ))
        scenario.commit()
        result = compute_cost_drag(_START, _END, scenario)
        assert result["total_cost_usd"] == pytest.approx(6.30 + 3.075, rel=1e-9)


# ---------------------------------------------------------------------------
# reconcile_attribution
# ---------------------------------------------------------------------------


class TestReconcileAttribution:
    def test_unexplained_surfaced_explicitly(self) -> None:
        """Gap must appear in the returned dict, never silently dropped."""
        result = reconcile_attribution(
            total_return=0.05,
            sector_contributions={"XLK": 0.03, "XLF": 0.02, "CASH": 0.0},
            cost_drag_bps=0.0,
        )
        assert "unexplained_pct" in result

    def test_perfect_reconciliation_gives_zero_gap(self) -> None:
        result = reconcile_attribution(
            total_return=0.05,
            sector_contributions={"XLK": 0.05, "CASH": 0.0},
            cost_drag_bps=0.0,
        )
        assert result["unexplained_pct"] == pytest.approx(0.0, abs=1e-12)

    def test_cost_drag_reduces_explained_return(self) -> None:
        # 10 bps drag = 0.001 fraction; contributions sum to 0.051
        result = reconcile_attribution(
            total_return=0.05,
            sector_contributions={"XLK": 0.051, "CASH": 0.0},
            cost_drag_bps=10.0,
        )
        assert result["cost_drag_fraction"] == pytest.approx(0.001)
        assert result["explained"] == pytest.approx(0.051 - 0.001)
        # unexplained = 0.05 − 0.05 = 0.0
        assert result["unexplained_pct"] == pytest.approx(0.0, abs=1e-12)

    def test_reconcile_scenario_gap_within_tolerance(self, scenario: Session) -> None:
        """Full pipeline: gap from the hand-calculated scenario < 0.5%."""
        total_return = compute_period_return(_START, _END, scenario)["total_return_pct"]
        contribs = compute_sector_contribution(_START, _END, scenario, _PRICES)
        cost_drag = compute_cost_drag(_START, _END, scenario)["cost_drag_bps"]
        result = reconcile_attribution(total_return, contribs, cost_drag)

        # Within 0.5% absolute — the weight-averaging approximation is the source
        assert abs(result["unexplained_pct"]) < 0.005

    def test_result_keys_complete(self) -> None:
        result = reconcile_attribution(0.05, {"XLK": 0.05}, 0.0)
        assert set(result.keys()) == {
            "total_return", "sum_contributions", "cost_drag_fraction",
            "explained", "unexplained_pct",
        }
