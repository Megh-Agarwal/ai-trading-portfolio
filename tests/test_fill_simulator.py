"""Tests for src/execution/fill_simulator.py — Ticket 4.3."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from config import TransactionCostsConfig
from db.models import Base, Trade
from exceptions import NegativeCashError
from execution.fill_simulator import apply_fills_to_state, simulate_all_fills, simulate_fill
from execution.orders import Order


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _config(spread_bps: float = 1.0, slippage_bps: float = 2.0) -> TransactionCostsConfig:
    """3 bps total by default, matching the acceptance-criteria examples."""
    return TransactionCostsConfig(
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        min_trade_threshold=0.001,
    )


def _buy(ticker: str, shares: int, price: float) -> Order:
    return Order(
        ticker=ticker, side="buy", shares=shares,
        estimated_price=price, estimated_value=shares * price, reason="test",
    )


def _sell(ticker: str, shares: int, price: float) -> Order:
    return Order(
        ticker=ticker, side="sell", shares=shares,
        estimated_price=price, estimated_value=shares * price, reason="test",
    )


# ---------------------------------------------------------------------------
# simulate_fill — unit tests
# ---------------------------------------------------------------------------


class TestSimulateFill:
    def test_buy_fill_gross_net_values(self) -> None:
        """Buy: net_value = gross_value + cost_usd (costs buyer more)."""
        fill = simulate_fill(_buy("XLK", 100, 200.0), _config())
        assert fill["gross_value"] == pytest.approx(20_000.0)
        assert fill["cost_usd"] == pytest.approx(6.0)       # 20_000 × 3bps
        assert fill["net_value"] == pytest.approx(20_006.0)

    def test_sell_fill_gross_net_values(self) -> None:
        """Sell: net_value = gross_value − cost_usd (seller receives less)."""
        fill = simulate_fill(_sell("XLK", 100, 200.0), _config())
        assert fill["gross_value"] == pytest.approx(20_000.0)
        assert fill["cost_usd"] == pytest.approx(6.0)
        assert fill["net_value"] == pytest.approx(19_994.0)

    def test_fill_price_equals_estimated_price(self) -> None:
        fill = simulate_fill(_buy("XLF", 50, 40.0), _config())
        assert fill["fill_price"] == pytest.approx(40.0)

    def test_fill_dict_keys(self) -> None:
        fill = simulate_fill(_buy("XLK", 10, 100.0), _config())
        assert set(fill.keys()) == {"ticker", "side", "shares", "fill_price",
                                    "gross_value", "cost_usd", "net_value"}

    def test_zero_bps_fill_has_zero_cost(self) -> None:
        fill = simulate_fill(_buy("XLK", 100, 200.0), _config(0.0, 0.0))
        assert fill["cost_usd"] == pytest.approx(0.0)
        assert fill["net_value"] == pytest.approx(fill["gross_value"])


# ---------------------------------------------------------------------------
# Mandatory acceptance-criteria tests
# ---------------------------------------------------------------------------


class TestBuyFillUpdatesPositionAndCashCorrectly:
    def test_buy_fill_updates_position_and_cash_correctly(self, db: Session) -> None:
        """100 XLK @ $200, 3bps → position +100, CASH −$20,006."""
        order = _buy("XLK", 100, 200.0)
        fills = simulate_all_fills([order], "2024-01-05", db, _config())

        positions = apply_fills_to_state(
            "2024-01-05", fills,
            current_positions={"CASH": 100_000.0, "XLK": 0.0},
            db=db,
        )

        assert positions["XLK"] == pytest.approx(100.0)
        assert positions["CASH"] == pytest.approx(100_000.0 - 20_006.0)

    def test_buy_writes_trade_record(self, db: Session) -> None:
        order = _buy("XLK", 100, 200.0)
        simulate_all_fills([order], "2024-01-05", db, _config())

        trade = db.execute(select(Trade).where(Trade.ticker == "XLK")).scalar_one()
        assert trade.side == "buy"
        assert trade.shares == pytest.approx(100.0)
        assert trade.price == pytest.approx(200.0)
        assert trade.commission == pytest.approx(6.0)
        assert trade.slippage == pytest.approx(0.0)


class TestSellFillUpdatesPositionAndCashCorrectly:
    def test_sell_fill_updates_position_and_cash_correctly(self, db: Session) -> None:
        """100 XLK @ $200, 3bps → position −100, CASH +$19,994."""
        order = _sell("XLK", 100, 200.0)
        fills = simulate_all_fills([order], "2024-01-05", db, _config())

        positions = apply_fills_to_state(
            "2024-01-05", fills,
            current_positions={"CASH": 0.0, "XLK": 100.0},
            db=db,
        )

        assert positions["XLK"] == pytest.approx(0.0)
        assert positions["CASH"] == pytest.approx(19_994.0)

    def test_sell_writes_trade_record(self, db: Session) -> None:
        order = _sell("XLK", 100, 200.0)
        simulate_all_fills([order], "2024-01-05", db, _config())

        trade = db.execute(select(Trade).where(Trade.ticker == "XLK")).scalar_one()
        assert trade.side == "sell"
        assert trade.commission == pytest.approx(6.0)


class TestRoundTripCostsChargedTwice:
    def test_round_trip_costs_charged_twice(self, db: Session) -> None:
        """Buy then sell back at same price → position unchanged, CASH −2×cost."""
        start_cash = 100_000.0

        # --- buy leg ---
        buy_fills = simulate_all_fills([_buy("XLK", 100, 200.0)], "2024-01-05", db, _config())
        positions = apply_fills_to_state(
            "2024-01-05", buy_fills,
            current_positions={"CASH": start_cash, "XLK": 0.0},
            db=db,
        )

        # --- sell leg ---
        sell_fills = simulate_all_fills([_sell("XLK", 100, 200.0)], "2024-01-06", db, _config())
        positions = apply_fills_to_state(
            "2024-01-06", sell_fills,
            current_positions=positions,
            db=db,
        )

        assert positions["XLK"] == pytest.approx(0.0)
        # paid $6 buying, received $6 less selling → net −$12
        assert positions["CASH"] == pytest.approx(start_cash - 12.0)

    def test_round_trip_cash_reduction_equals_2x_cost(self, db: Session) -> None:
        buy_fill = simulate_fill(_buy("XLK", 100, 200.0), _config())
        sell_fill = simulate_fill(_sell("XLK", 100, 200.0), _config())
        # cash outflow for buy + cash reduction from sell == 2 × cost_usd
        total_cost = buy_fill["cost_usd"] + sell_fill["cost_usd"]
        assert total_cost == pytest.approx(12.0)


class TestNegativeCashRaisesError:
    def test_negative_cash_raises_error(self, db: Session) -> None:
        """Buying more than available cash raises NegativeCashError."""
        order = _buy("XLK", 100, 200.0)  # costs 20_006
        fills = simulate_all_fills([order], "2024-01-05", db, _config())

        with pytest.raises(NegativeCashError):
            apply_fills_to_state(
                "2024-01-05", fills,
                current_positions={"CASH": 5_000.0, "XLK": 0.0},
                db=db,
            )

    def test_negative_cash_does_not_write_positions(self, db: Session) -> None:
        """On NegativeCashError, write_positions must not have committed."""
        from db.models import Position
        order = _buy("XLK", 100, 200.0)
        fills = simulate_all_fills([order], "2024-01-05", db, _config())

        with pytest.raises(NegativeCashError):
            apply_fills_to_state(
                "2024-01-05", fills,
                current_positions={"CASH": 5_000.0},
                db=db,
            )

        # No positions row should have been written
        count = db.execute(
            select(Position).where(Position.date == "2024-01-05")
        ).scalars().all()
        assert count == []


# ---------------------------------------------------------------------------
# simulate_all_fills — additional coverage
# ---------------------------------------------------------------------------


class TestSimulateAllFills:
    def test_multiple_orders_write_multiple_trade_rows(self, db: Session) -> None:
        orders = [_buy("XLK", 50, 200.0), _sell("XLF", 30, 40.0)]
        fills = simulate_all_fills(orders, "2024-01-05", db, _config())

        assert len(fills) == 2
        trades = db.execute(select(Trade)).scalars().all()
        assert len(trades) == 2

    def test_empty_orders_returns_empty_fills(self, db: Session) -> None:
        fills = simulate_all_fills([], "2024-01-05", db, _config())
        assert fills == []

    def test_fills_preserve_order_sequence(self, db: Session) -> None:
        orders = [_buy("XLK", 50, 200.0), _buy("XLF", 100, 40.0)]
        fills = simulate_all_fills(orders, "2024-01-05", db, _config())
        assert fills[0]["ticker"] == "XLK"
        assert fills[1]["ticker"] == "XLF"
