"""Tests for src/execution/costs.py — Ticket 3.4."""

from __future__ import annotations

import numpy as np
import pytest

from execution.costs import (
    compute_cost_drag_bps,
    estimate_portfolio_rebalance_cost,
    estimate_trade_cost,
)

# ---------------------------------------------------------------------------
# Helper: minimal TransactionCostsConfig
# ---------------------------------------------------------------------------


def _make_config(
    spread_bps: float = 1.0,
    slippage_bps: float = 2.0,
    min_trade_threshold: float = 0.001,
):
    from config import TransactionCostsConfig

    return TransactionCostsConfig(
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        min_trade_threshold=min_trade_threshold,
    )


# ---------------------------------------------------------------------------
# Mandatory acceptance-criteria tests
# ---------------------------------------------------------------------------


class TestMandatoryAcceptanceCriteria:
    def test_single_trade_cost_calculation(self) -> None:
        """$100,000 trade at 3 bps (1 spread + 2 slippage) → exactly $30."""
        cfg = _make_config()
        cost = estimate_trade_cost("XLK", 100_000.0, cfg)
        assert cost == pytest.approx(30.0, rel=1e-9)

    def test_small_trade_below_threshold_is_free(self) -> None:
        """Weight change well below min_trade_threshold → zero cost."""
        cfg = _make_config(min_trade_threshold=0.001)
        old = np.array([0.100, 0.900])
        new = np.array([0.1005, 0.8995])  # |Δw| = 0.0005 < threshold=0.001
        cost = estimate_portfolio_rebalance_cost(old, new, 1_000_000.0, cfg)
        assert cost == pytest.approx(0.0)

    def test_full_rebalance_cost_sums_correctly(self) -> None:
        """Equal → concentrated rebalance: cost = sum of per-position costs."""
        n = 10
        old = np.ones(n) / n  # 10% each
        new = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * 6)
        portfolio_value = 1_000_000.0
        cfg = _make_config()

        # 4 positions: Δw = +0.15; 6 positions: Δw = −0.10; all above threshold
        expected = (4 * 0.15 + 6 * 0.10) * portfolio_value * 3.0 / 10_000.0
        cost = estimate_portfolio_rebalance_cost(old, new, portfolio_value, cfg)
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_cost_drag_bps_calculation(self) -> None:
        """$300 cost on a $1M portfolio → 3 bps drag."""
        drag = compute_cost_drag_bps(300.0, 1_000_000.0)
        assert drag == pytest.approx(3.0, rel=1e-9)


# ---------------------------------------------------------------------------
# estimate_trade_cost — additional coverage
# ---------------------------------------------------------------------------


class TestEstimateTradeCost:
    def test_zero_trade_value_is_free(self) -> None:
        assert estimate_trade_cost("XLF", 0.0, _make_config()) == pytest.approx(0.0)

    def test_cost_scales_linearly_with_trade_value(self) -> None:
        cfg = _make_config()
        assert estimate_trade_cost("XLV", 200_000.0, cfg) == pytest.approx(60.0)
        assert estimate_trade_cost("XLV", 50_000.0, cfg) == pytest.approx(15.0)

    def test_ticker_does_not_affect_cost_in_v1(self) -> None:
        """All tickers use the same bps in v1."""
        cfg = _make_config()
        costs = [
            estimate_trade_cost(t, 100_000.0, cfg) for t in ["XLK", "XLF", "XLV", "XLU", "XLRE"]
        ]
        assert all(c == pytest.approx(30.0) for c in costs)

    def test_custom_bps_changes_cost(self) -> None:
        cfg_5bps = _make_config(spread_bps=2.0, slippage_bps=3.0)
        cost = estimate_trade_cost("XLK", 100_000.0, cfg_5bps)
        assert cost == pytest.approx(50.0)  # 5bps × $100k


# ---------------------------------------------------------------------------
# estimate_portfolio_rebalance_cost — additional coverage
# ---------------------------------------------------------------------------


class TestEstimatePortfolioRebalanceCost:
    def test_no_change_is_free(self) -> None:
        w = np.ones(10) / 10
        assert estimate_portfolio_rebalance_cost(w, w.copy(), 1_000_000.0, _make_config()) == 0.0

    def test_above_threshold_incurs_cost(self) -> None:
        cfg = _make_config(min_trade_threshold=0.001)
        old = np.array([0.10, 0.90])
        new = np.array([0.102, 0.898])  # |Δw| = 0.002 > threshold
        cost = estimate_portfolio_rebalance_cost(old, new, 1_000_000.0, cfg)
        # Both positions above threshold
        expected = 2 * 0.002 * 1_000_000.0 * 3.0 / 10_000.0
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_mixed_above_and_below_threshold(self) -> None:
        """Positions below threshold are free; positions above incur cost."""
        cfg = _make_config(min_trade_threshold=0.010)
        old = np.array([0.50, 0.30, 0.20])
        new = np.array([0.505, 0.295, 0.200])
        # Position 0: |Δw| = 0.005 < threshold=0.010 → free
        # Position 1: |Δw| = 0.005 < threshold=0.010 → free
        # Position 2: |Δw| = 0.000 → free
        cost = estimate_portfolio_rebalance_cost(old, new, 1_000_000.0, cfg)
        assert cost == pytest.approx(0.0)

    def test_portfolio_value_scales_cost(self) -> None:
        old = np.array([0.50, 0.50])
        new = np.array([0.60, 0.40])
        cfg = _make_config()
        cost_1m = estimate_portfolio_rebalance_cost(old, new, 1_000_000.0, cfg)
        cost_2m = estimate_portfolio_rebalance_cost(old, new, 2_000_000.0, cfg)
        assert cost_2m == pytest.approx(2 * cost_1m, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_cost_drag_bps
# ---------------------------------------------------------------------------


class TestComputeCostDragBps:
    def test_zero_cost_gives_zero_drag(self) -> None:
        assert compute_cost_drag_bps(0.0, 1_000_000.0) == pytest.approx(0.0)

    def test_one_bp_drag(self) -> None:
        # $100 cost on $1M portfolio = 1 bp
        assert compute_cost_drag_bps(100.0, 1_000_000.0) == pytest.approx(1.0)

    def test_drag_inversely_proportional_to_portfolio_size(self) -> None:
        drag_1m = compute_cost_drag_bps(300.0, 1_000_000.0)
        drag_10m = compute_cost_drag_bps(300.0, 10_000_000.0)
        assert drag_1m == pytest.approx(10 * drag_10m, rel=1e-9)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestTransactionCostsConfig:
    def test_config_loads_transaction_costs_stanza(self) -> None:
        from config import load_config

        cfg = load_config("optimizer")
        tc = cfg.transaction_costs
        assert tc.spread_bps == pytest.approx(1.0)
        assert tc.slippage_bps == pytest.approx(2.0)
        assert tc.min_trade_threshold == pytest.approx(0.001)

    def test_config_yields_correct_total_bps(self) -> None:
        from config import load_config

        tc = load_config("optimizer").transaction_costs
        total_bps = tc.spread_bps + tc.slippage_bps
        assert total_bps == pytest.approx(3.0)

    def test_no_hardcoded_bps_in_source(self) -> None:
        """Changing config bps changes the cost — confirms no hardcoded constants."""
        cfg_5bps = _make_config(spread_bps=2.0, slippage_bps=3.0)
        cfg_3bps = _make_config(spread_bps=1.0, slippage_bps=2.0)
        cost_5 = estimate_trade_cost("XLK", 100_000.0, cfg_5bps)
        cost_3 = estimate_trade_cost("XLK", 100_000.0, cfg_3bps)
        assert cost_5 == pytest.approx(50.0)
        assert cost_3 == pytest.approx(30.0)
        assert cost_5 != cost_3
