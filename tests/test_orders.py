"""Tests for src/execution/orders.py — Ticket 4.2."""
from __future__ import annotations

import pytest

from execution.orders import Order, generate_orders, validate_orders_affordable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prices(**kwargs: float) -> dict[str, float]:
    return dict(kwargs)


def _positions(**kwargs: float) -> dict[str, float]:
    return dict(kwargs)


def _weights(**kwargs: float) -> dict[str, float]:
    return dict(kwargs)


def _find(orders: list[Order], ticker: str) -> Order:
    matches = [o for o in orders if o.ticker == ticker]
    assert len(matches) == 1, f"expected exactly 1 order for {ticker}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Mandatory acceptance-criteria tests
# ---------------------------------------------------------------------------


class TestGeneratesBuyForNewPosition:
    def test_generates_buy_for_new_position(self) -> None:
        """Ticker in target_weights but not held → buy order."""
        orders = generate_orders(
            target_weights=_weights(XLK=0.15),
            current_positions=_positions(CASH=100_000.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert len(orders) == 1
        o = orders[0]
        assert o.ticker == "XLK"
        assert o.side == "buy"
        assert o.shares == 150  # floor(15_000 / 100)
        assert o.estimated_price == pytest.approx(100.0)
        assert o.estimated_value == pytest.approx(15_000.0)

    def test_buy_reason_shows_target_vs_current(self) -> None:
        orders = generate_orders(
            target_weights=_weights(XLK=0.20),
            current_positions=_positions(CASH=100_000.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert "20.0%" in orders[0].reason
        assert "0.0%" in orders[0].reason


class TestGeneratesSellForDroppedPosition:
    def test_generates_sell_for_dropped_position(self) -> None:
        """Ticker held but not in target_weights → full sell order."""
        orders = generate_orders(
            target_weights={},  # XLK no longer wanted
            current_positions=_positions(CASH=80_000.0, XLK=200.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert len(orders) == 1
        o = orders[0]
        assert o.ticker == "XLK"
        assert o.side == "sell"
        assert o.shares == 200  # floor(20_000 / 100)
        assert o.estimated_value == pytest.approx(20_000.0)

    def test_partial_sell_for_reduced_weight(self) -> None:
        """Reducing a position from 20% to 10% → sell half the shares."""
        orders = generate_orders(
            target_weights=_weights(XLK=0.10),
            current_positions=_positions(CASH=80_000.0, XLK=200.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        o = _find(orders, "XLK")
        assert o.side == "sell"
        assert o.shares == 100  # reduce from 200 → 100 shares


class TestNoOrdersWhenAlreadyAtTarget:
    def test_no_orders_when_already_at_target(self) -> None:
        """Current weight matches target within min_trade_threshold → empty list."""
        orders = generate_orders(
            target_weights=_weights(XLK=0.15),
            current_positions=_positions(CASH=85_000.0, XLK=150.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert orders == []

    def test_tiny_deviation_below_threshold_no_order(self) -> None:
        """Deviation of 0.05% (< default 0.1% threshold) → no order."""
        # XLK at 15.05% vs target 15.00% → Δ = 0.0005 < 0.001 threshold
        orders = generate_orders(
            target_weights=_weights(XLK=0.1500),
            current_positions=_positions(CASH=84_950.0, XLK=150.5),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert orders == []

    def test_deviation_above_threshold_generates_order(self) -> None:
        """Deviation above threshold → at least one order."""
        # XLK at 10% vs target 15% → Δ = 5% > threshold
        orders = generate_orders(
            target_weights=_weights(XLK=0.15),
            current_positions=_positions(CASH=90_000.0, XLK=100.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert len(orders) == 1
        assert orders[0].side == "buy"


class TestRoundsDownNeverUp:
    def test_rounds_down_never_up(self) -> None:
        """Fractional share count is always floored, never rounded up."""
        # target_value = 10_350, price = 103 → 100.485... shares → floor = 100
        orders = generate_orders(
            target_weights=_weights(XLK=0.10350),
            current_positions=_positions(CASH=100_000.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=103.0),
        )
        o = _find(orders, "XLK")
        assert o.shares == 100  # not 101
        assert o.estimated_value == pytest.approx(100 * 103.0)

    def test_floor_means_value_never_exceeds_target(self) -> None:
        """Ordered value (shares × price) must be ≤ target_value."""
        for price in [97.3, 103.7, 201.5, 55.0]:
            orders = generate_orders(
                target_weights=_weights(XLK=0.12),
                current_positions=_positions(CASH=100_000.0),
                portfolio_value=100_000.0,
                prices=_prices(XLK=price),
            )
            if orders:
                target_value = 0.12 * 100_000.0
                assert orders[0].estimated_value <= target_value + 1e-9

    def test_zero_shares_after_floor_is_skipped(self) -> None:
        """If floor(delta_value / price) == 0 the ticker is skipped entirely."""
        # delta_value ≈ $50 (0.5% of 10k), price = $200 → floor(50/200) = 0
        orders = generate_orders(
            target_weights=_weights(XLK=0.105),
            current_positions=_positions(CASH=9_000.0, XLK=50.0),
            portfolio_value=10_000.0,
            prices=_prices(XLK=200.0),
            min_trade_threshold=0.001,
        )
        # Δvalue = 10_000×0.105 - 50×200 = 1_050 - 10_000 = -8_950 → sell
        # Actually let me check: current is 50 shares at 200 = 10_000 ≥ portfolio.
        # Let me use a case where delta is small:
        # target=0.100, current XLK=50 shares @ 200 → current_value=10_000=portfolio_value
        # Δvalue = 0, skip. Better:
        # portfolio=100_000, target=0.002 (0.2%), price=300 →
        # target_value=200, delta_shares=floor(200/300)=0 → skip
        orders2 = generate_orders(
            target_weights=_weights(XLK=0.002),
            current_positions=_positions(CASH=100_000.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=300.0),
            min_trade_threshold=0.001,
        )
        assert orders2 == []


class TestAffordabilityCheckCatchesInfeasibleSet:
    def test_affordability_check_catches_infeasible_set(self) -> None:
        """Buy total exceeds cash + sell proceeds → returns False."""
        orders = [
            Order("XLK", "buy", 100, 200.0, 20_000.0, "test"),
            Order("XLF", "sell", 50, 50.0, 2_500.0, "test"),
        ]
        # available = 10_000 + 2_500 = 12_500 < 20_000
        assert validate_orders_affordable(orders, available_cash=10_000.0) is False

    def test_affordability_passes_when_sells_cover_buys(self) -> None:
        """Sell proceeds bridge the gap → returns True."""
        orders = [
            Order("XLK", "buy", 100, 200.0, 20_000.0, "test"),
            Order("XLF", "sell", 300, 50.0, 15_000.0, "test"),
        ]
        # available = 6_000 + 15_000 = 21_000 >= 20_000
        assert validate_orders_affordable(orders, available_cash=6_000.0) is True

    def test_affordability_with_no_orders(self) -> None:
        assert validate_orders_affordable([], available_cash=0.0) is True

    def test_affordability_buys_only_within_cash(self) -> None:
        orders = [Order("XLK", "buy", 10, 100.0, 1_000.0, "test")]
        assert validate_orders_affordable(orders, available_cash=1_000.0) is True
        assert validate_orders_affordable(orders, available_cash=999.99) is False

    def test_affordability_sells_only_always_passes(self) -> None:
        orders = [Order("XLK", "sell", 50, 100.0, 5_000.0, "test")]
        assert validate_orders_affordable(orders, available_cash=0.0) is True


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


class TestGenerateOrdersMultipleTickers:
    def test_multiple_tickers_mix_of_buy_sell(self) -> None:
        """Increasing XLK, decreasing XLF, dropping XLE → 3 orders."""
        orders = generate_orders(
            target_weights=_weights(XLK=0.20, XLF=0.05),
            current_positions=_positions(CASH=70_000.0, XLK=100.0, XLF=100.0, XLE=50.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0, XLF=100.0, XLE=100.0),
        )
        tickers = {o.ticker for o in orders}
        assert "XLK" in tickers
        assert "XLF" in tickers
        assert "XLE" in tickers
        assert _find(orders, "XLK").side == "buy"
        assert _find(orders, "XLF").side == "sell"
        assert _find(orders, "XLE").side == "sell"

    def test_cash_ticker_never_appears_in_orders(self) -> None:
        """CASH is never included in the order list regardless of its weight."""
        orders = generate_orders(
            target_weights=_weights(CASH=0.10, XLK=0.90),
            current_positions=_positions(CASH=100_000.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
        )
        assert all(o.ticker != "CASH" for o in orders)

    def test_custom_min_trade_threshold(self) -> None:
        """Higher threshold suppresses small trades."""
        # Δ = 2% of portfolio — below a 5% threshold
        orders = generate_orders(
            target_weights=_weights(XLK=0.12),
            current_positions=_positions(CASH=90_000.0, XLK=100.0),
            portfolio_value=100_000.0,
            prices=_prices(XLK=100.0),
            min_trade_threshold=0.05,
        )
        assert orders == []

    def test_zero_portfolio_value_returns_empty(self) -> None:
        orders = generate_orders(
            target_weights=_weights(XLK=0.10),
            current_positions=_positions(CASH=0.0),
            portfolio_value=0.0,
            prices=_prices(XLK=100.0),
        )
        assert orders == []
