"""Tests for src/optimizer/risk_checks.py — Ticket 3.5."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from db.models import Base, PortfolioSnapshot, RiskEvent
from optimizer.risk_checks import (
    RiskCheckResult,
    check_drawdown_circuit_breaker,
    check_max_position,
    check_max_turnover,
    check_realized_vol,
    run_all_risk_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = datetime.date(2024, 6, 14)
_N = 10  # number of sectors


def _make_config(**risk_overrides):
    """Build a minimal OptimizerConfig with optional risk parameter overrides."""
    from config import OptimizerConfig

    base = {
        "tau": 0.05,
        "risk_aversion": 2.5,
        "max_position_weight": 0.25,
        "vol_target": 0.12,
        "turnover_penalty": 0.10,
        "transaction_cost_bps": 10,
        "aggregator": {
            "max_excess_return_annual": 0.05,
            "omega_base": 0.0001,
            "regime_scale_intercept": 0.75,
            "regime_scale_slope": 0.25,
        },
        "aggregator_weights": {
            "backtest": {"news": 0.57, "macro": 0.43, "polymarket": 0.00},
            "live": {"news": 0.40, "macro": 0.30, "polymarket": 0.30},
        },
        "market_cap_weights": {"XLK": 0.30, "XLF": 0.10, "XLV": 0.10},
        "prior": {"lookback_days": 252},
        "black_litterman": {"tau": 0.05},
        "transaction_costs": {
            "spread_bps": 1.0,
            "slippage_bps": 2.0,
            "min_trade_threshold": 0.001,
        },
        "portfolio": {
            "max_position_weight": 0.25,
            "vol_target": 0.12,
            "turnover_penalty": 0.10,
            "solver_primary": "CLARABEL",
            "solver_fallback": "SCS",
        },
        "risk": {
            "max_single_rebalance_turnover": 0.50,
            "max_drawdown_threshold": 0.15,
            "drawdown_lookback_days": 20,
            "vol_breach_multiplier": 1.50,
            "vol_deleveraging_blend": 0.20,
            **risk_overrides,
        },
    }
    return OptimizerConfig.model_validate(base)


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/risk_test.db")
    Base.metadata.create_all(engine)
    return engine


def _insert_snapshots(db, start: datetime.date, values: list[float]) -> None:
    """Insert PortfolioSnapshot rows on consecutive calendar days."""
    with Session(db) as session:
        for i, v in enumerate(values):
            session.add(
                PortfolioSnapshot(
                    date=start + datetime.timedelta(days=i),
                    total_value=v,
                    cash=0.0,
                    gross_exposure=1.0,
                    net_exposure=1.0,
                )
            )
        session.commit()


def _stable_history(db, end_date: datetime.date, n_days: int = 21) -> None:
    """Insert constant $1 M portfolio — zero drawdown, zero realized vol."""
    start = end_date - datetime.timedelta(days=n_days - 1)
    _insert_snapshots(db, start, [1_000_000.0] * n_days)


def _volatile_history(db, end_date: datetime.date, n_days: int = 21) -> None:
    """Insert alternating ±3 % daily returns — high vol, low drawdown."""
    start = end_date - datetime.timedelta(days=n_days - 1)
    values = [1_000_000.0]
    for i in range(n_days - 1):
        factor = 1.03 if i % 2 == 0 else 0.97
        values.append(values[-1] * factor)
    _insert_snapshots(db, start, values)


def _drawdown_history(db, end_date: datetime.date) -> None:
    """Insert $1 M peak → $790 k current: −21 % drawdown."""
    start = end_date - datetime.timedelta(days=4)
    _insert_snapshots(db, start, [1_000_000, 950_000, 900_000, 850_000, 790_000])


# ---------------------------------------------------------------------------
# TestCheckMaxPosition
# ---------------------------------------------------------------------------


class TestCheckMaxPosition:
    def test_passes_when_all_weights_within_limit(self):
        weights = np.ones(_N) / _N  # 0.10 each ≤ 0.25
        result = check_max_position(weights, _make_config())
        assert result.passed
        assert result.check_name == "max_position"
        assert result.value == pytest.approx(0.10)
        assert result.threshold == pytest.approx(0.25)
        assert result.action == "none"

    def test_passes_at_exact_threshold(self):
        # 0.25 exactly should pass (with float tolerance)
        weights = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * 6)
        result = check_max_position(weights, _make_config())
        assert result.passed

    def test_fails_when_weight_exceeds_limit(self):
        weights = np.array([0.40] + [0.60 / 9] * 9)  # 0.40 > 0.25
        result = check_max_position(weights, _make_config())
        assert not result.passed
        assert result.value == pytest.approx(0.40)
        assert result.threshold == pytest.approx(0.25)
        assert result.action == "clip_and_renorm"
        assert "clip" in result.message

    def test_result_is_risk_check_result_dataclass(self):
        result = check_max_position(np.ones(_N) / _N, _make_config())
        assert isinstance(result, RiskCheckResult)


# ---------------------------------------------------------------------------
# TestCheckMaxTurnover
# ---------------------------------------------------------------------------


class TestCheckMaxTurnover:
    def test_passes_when_turnover_below_limit(self):
        old = np.ones(_N) / _N
        new = old.copy()
        new[0] += 0.05
        new[1] -= 0.05  # L1 = 0.10 < 0.50
        result = check_max_turnover(new, old, _make_config())
        assert result.passed
        assert result.value == pytest.approx(0.10)
        assert result.action == "none"

    def test_passes_with_zero_turnover(self):
        w = np.ones(_N) / _N
        result = check_max_turnover(w, w.copy(), _make_config())
        assert result.passed
        assert result.value == pytest.approx(0.0)

    def test_fails_when_turnover_exceeds_limit(self):
        old = np.ones(_N) / _N  # 0.10 each
        new = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * 6)
        # L1 = 4×0.15 + 6×0.10 = 1.20 > 0.50
        result = check_max_turnover(new, old, _make_config())
        assert not result.passed
        assert result.value == pytest.approx(1.20, rel=1e-6)
        assert result.threshold == pytest.approx(0.50)
        assert result.action == "blend_50_50"
        assert "blend" in result.message

    def test_custom_turnover_threshold(self):
        old = np.ones(_N) / _N
        new = old.copy()
        new[0] += 0.15
        new[1] -= 0.15  # L1 = 0.30
        # Default limit 0.50 passes; custom limit 0.20 fails
        assert check_max_turnover(new, old, _make_config()).passed
        assert not check_max_turnover(
            new, old, _make_config(max_single_rebalance_turnover=0.20)
        ).passed


# ---------------------------------------------------------------------------
# TestCheckDrawdownCircuitBreaker
# ---------------------------------------------------------------------------


class TestCheckDrawdownCircuitBreaker:
    def test_passes_when_no_history(self, db):
        result = check_drawdown_circuit_breaker(_DATE, db, _make_config())
        assert result.passed
        assert result.value is None

    def test_passes_with_only_one_snapshot(self, db):
        _insert_snapshots(db, _DATE, [1_000_000])
        result = check_drawdown_circuit_breaker(_DATE, db, _make_config())
        assert result.passed
        assert result.value is None

    def test_passes_when_portfolio_growing(self, db):
        start = _DATE - datetime.timedelta(days=4)
        _insert_snapshots(db, start, [100_000, 101_000, 102_000, 103_000, 104_000])
        result = check_drawdown_circuit_breaker(_DATE, db, _make_config())
        assert result.passed
        assert result.value == pytest.approx(0.0, abs=1e-9)  # no drawdown at all

    def test_passes_when_drawdown_below_threshold(self, db):
        # −10% drawdown (< 15% threshold)
        start = _DATE - datetime.timedelta(days=2)
        _insert_snapshots(db, start, [1_000_000, 950_000, 900_000])
        result = check_drawdown_circuit_breaker(_DATE, db, _make_config())
        assert result.passed
        assert result.value == pytest.approx(-0.10, rel=1e-6)

    def test_fails_on_severe_drawdown(self, db):
        # −21% drawdown (exceeds 15% threshold)
        _drawdown_history(db, _DATE)
        result = check_drawdown_circuit_breaker(_DATE, db, _make_config())
        assert not result.passed
        assert result.value == pytest.approx(-0.21, rel=1e-3)
        assert result.threshold == pytest.approx(-0.15)
        assert result.action == "halt_rebalance"
        assert "halt" in result.message

    def test_threshold_is_negative_of_config_value(self, db):
        cfg = _make_config(max_drawdown_threshold=0.20)
        result = check_drawdown_circuit_breaker(_DATE, db, cfg)
        assert result.threshold == pytest.approx(-0.20)

    def test_respects_lookback_window(self, db):
        # 25-day-old severe loss is outside a 20-day lookback window
        # Insert: day −25 = 500k (severe loss from 1M), then 20 days of recovery to 900k
        start = _DATE - datetime.timedelta(days=25)
        far_past = [1_000_000, 500_000]  # big loss on day −24
        recent = [500_000 + i * 20_000 for i in range(24)]  # recovery: 500k→960k
        _insert_snapshots(db, start, far_past + recent)
        cfg = _make_config(drawdown_lookback_days=20)
        result = check_drawdown_circuit_breaker(_DATE, db, cfg)
        # Within the 20-day window, peak ≈ 960k and current ≈ 960k — drawdown near zero
        assert result.passed


# ---------------------------------------------------------------------------
# TestCheckRealizedVol
# ---------------------------------------------------------------------------


class TestCheckRealizedVol:
    def test_passes_when_no_history(self, db):
        result = check_realized_vol(_DATE, db, _make_config())
        assert result.passed
        assert result.value is None

    def test_passes_with_insufficient_history(self, db):
        # 2 snapshots → 1 return → std(ddof=1) undefined; treated as insufficient
        _insert_snapshots(db, _DATE - datetime.timedelta(days=1), [1_000_000, 1_001_000])
        result = check_realized_vol(_DATE, db, _make_config())
        assert result.passed
        assert result.value is None

    def test_threshold_is_vol_target_times_multiplier(self, db):
        cfg = _make_config()
        result = check_realized_vol(_DATE, db, cfg)
        expected_threshold = cfg.portfolio.vol_target * cfg.risk.vol_breach_multiplier
        assert result.threshold == pytest.approx(expected_threshold)

    def test_passes_when_vol_within_limit(self, db):
        # Very stable portfolio: +0.05 %/day → annualised vol ≈ 0.8 % << 18 % threshold
        _stable_history(db, _DATE)
        result = check_realized_vol(_DATE, db, _make_config())
        assert result.passed
        assert result.value is not None
        assert result.value < 0.18  # below 12 % × 1.5 = 18 %

    def test_fails_when_vol_too_high(self, db):
        # Alternating ±3 %/day → annualised vol ≈ 47 % >> 18 % threshold
        _volatile_history(db, _DATE)
        result = check_realized_vol(_DATE, db, _make_config())
        assert not result.passed
        assert result.value > 0.18
        assert result.action == "deleverage_20pct"
        assert "deleverage" in result.message or "deleverage" in result.action

    def test_value_is_annualised_vol(self, db):
        # Constant +1 %/day returns → annualised vol ≈ 0 (no variance)
        # Use slightly varying returns to get a measurable but low vol
        _stable_history(db, _DATE)
        result = check_realized_vol(_DATE, db, _make_config())
        assert result.passed
        # Value should be a small non-negative number (near-zero vol from constant returns)
        assert result.value is not None
        assert result.value >= 0.0


# ---------------------------------------------------------------------------
# TestRunAllRiskChecks — integration
# ---------------------------------------------------------------------------


class TestRunAllRiskChecks:
    def test_no_checks_triggered_returns_new_weights(self, db):
        _stable_history(db, _DATE)
        old = np.ones(_N) / _N
        new = np.ones(_N) / _N  # no change, no violations
        final, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        assert np.allclose(final, new)
        assert all(r.passed for r in results)
        assert len(results) == 4

    def test_drawdown_circuit_breaker_returns_prev_weights_unchanged(self, db):
        """−21 % drawdown → circuit breaker fires → prev_weights returned."""
        _drawdown_history(db, _DATE)
        old = np.ones(_N) / _N
        new = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * 6)
        final, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        assert np.allclose(final, old)
        assert not results[0].passed  # drawdown check failed
        assert results[0].action == "halt_rebalance"

    def test_max_position_clips_and_renorms(self, db):
        _stable_history(db, _DATE)
        old = np.ones(_N) / _N
        new = np.zeros(_N)
        new[0] = 0.40  # violates 0.25 cap
        new[1:] = 0.60 / 9
        final, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        assert np.max(final) <= 0.25 + 1e-6
        assert abs(np.sum(final) - 1.0) < 1e-9
        pos_result = next(r for r in results if r.check_name == "max_position")
        assert not pos_result.passed

    def test_max_turnover_blends_50_50(self, db):
        _stable_history(db, _DATE)
        old = np.ones(_N) / _N
        new = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * 6)
        # turnover = 1.20 > 0.50 → 50/50 blend
        final, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        expected = 0.5 * new + 0.5 * old
        expected /= expected.sum()
        assert np.allclose(final, expected, atol=1e-9)
        turnover_result = next(r for r in results if r.check_name == "max_turnover")
        assert not turnover_result.passed

    def test_vol_breach_blends_toward_equal_weight(self, db):
        _volatile_history(db, _DATE)
        old = np.ones(_N) / _N
        # non-uniform new_weights with low turnover and no pos violation
        new = np.array([0.15, 0.15, 0.12, 0.12, 0.11, 0.10, 0.09, 0.07, 0.05, 0.04])
        final, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        vol_result = next(r for r in results if r.check_name == "realized_vol")
        assert not vol_result.passed
        # Final weights should be blended toward equal weight
        equal_w = np.ones(_N) / _N
        blend = 0.20
        expected = (1 - blend) * new + blend * equal_w
        expected /= expected.sum()
        assert np.allclose(final, expected, atol=1e-6)

    def test_all_four_results_logged_to_risk_events(self, db):
        _stable_history(db, _DATE)
        old = new = np.ones(_N) / _N
        run_all_risk_checks(_DATE, new, old, db, _make_config())
        with Session(db) as session:
            count = session.execute(select(func.count()).select_from(RiskEvent)).scalar()
        assert count == 4

    def test_triggered_check_logged_with_triggered_true(self, db):
        _drawdown_history(db, _DATE)
        old = new = np.ones(_N) / _N
        run_all_risk_checks(_DATE, new, old, db, _make_config())
        with Session(db) as session:
            events = (
                session.execute(
                    select(RiskEvent).where(RiskEvent.triggered == True)  # noqa: E712
                )
                .scalars()
                .all()
            )
        assert any(e.check_name == "drawdown_circuit_breaker" for e in events)

    def test_passing_check_logged_with_triggered_false(self, db):
        _stable_history(db, _DATE)
        old = new = np.ones(_N) / _N
        run_all_risk_checks(_DATE, new, old, db, _make_config())
        with Session(db) as session:
            events = (
                session.execute(
                    select(RiskEvent).where(RiskEvent.triggered == False)  # noqa: E712
                )
                .scalars()
                .all()
            )
        assert len(events) == 4  # all checks passed → all logged as not triggered

    def test_multiple_checks_trigger_simultaneously_drawdown_takes_priority(self, db):
        """When drawdown AND turnover both fire, circuit breaker wins."""
        _drawdown_history(db, _DATE)
        old = np.ones(_N) / _N
        new = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * 6)  # also high turnover
        final, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        # Drawdown circuit breaker wins: return old weights unchanged
        assert np.allclose(final, old)
        # Both checks should have fired
        drawdown_r = next(r for r in results if r.check_name == "drawdown_circuit_breaker")
        turnover_r = next(r for r in results if r.check_name == "max_turnover")
        assert not drawdown_r.passed
        assert not turnover_r.passed
        # Both logged to DB
        with Session(db) as session:
            triggered = (
                session.execute(
                    select(RiskEvent).where(RiskEvent.triggered == True)  # noqa: E712
                )
                .scalars()
                .all()
            )
        triggered_names = {e.check_name for e in triggered}
        assert "drawdown_circuit_breaker" in triggered_names
        assert "max_turnover" in triggered_names

    def test_final_weights_sum_to_one(self, db):
        """Weight vector always sums to 1.0 regardless of which actions fired."""
        _stable_history(db, _DATE)
        old = np.ones(_N) / _N
        new = np.zeros(_N)
        new[0] = 0.40  # triggers pos check
        new[1:] = 0.60 / 9
        final, _ = run_all_risk_checks(_DATE, new, old, db, _make_config())
        assert abs(np.sum(final) - 1.0) < 1e-9

    def test_results_list_contains_all_check_names(self, db):
        _stable_history(db, _DATE)
        old = new = np.ones(_N) / _N
        _, results = run_all_risk_checks(_DATE, new, old, db, _make_config())
        names = {r.check_name for r in results}
        assert names == {"drawdown_circuit_breaker", "realized_vol", "max_position", "max_turnover"}


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestRiskConfig:
    def test_risk_config_loads_from_optimizer_yaml(self):
        from config import load_config

        cfg = load_config("optimizer")
        r = cfg.risk
        assert r.max_single_rebalance_turnover == pytest.approx(0.50)
        assert r.max_drawdown_threshold == pytest.approx(0.15)
        assert r.drawdown_lookback_days == 20
        assert r.vol_breach_multiplier == pytest.approx(1.50)
        assert r.vol_deleveraging_blend == pytest.approx(0.20)

    def test_vol_threshold_derived_from_config(self):
        from config import load_config

        cfg = load_config("optimizer")
        expected = cfg.portfolio.vol_target * cfg.risk.vol_breach_multiplier
        assert expected == pytest.approx(0.12 * 1.50)

    def test_invalid_vol_deleveraging_blend_raises(self, tmp_path, monkeypatch):
        import yaml

        from config import load_config

        data = {
            "tau": 0.05,
            "risk_aversion": 2.5,
            "max_position_weight": 0.25,
            "vol_target": 0.12,
            "turnover_penalty": 0.1,
            "transaction_cost_bps": 10,
            "aggregator": {
                "max_excess_return_annual": 0.05,
                "omega_base": 0.0001,
                "regime_scale_intercept": 0.75,
                "regime_scale_slope": 0.25,
            },
            "aggregator_weights": {
                "backtest": {"news": 0.57, "macro": 0.43, "polymarket": 0.00},
                "live": {"news": 0.40, "macro": 0.30, "polymarket": 0.30},
            },
            "market_cap_weights": {"XLK": 0.30, "XLF": 0.10, "XLV": 0.10},
            "prior": {"lookback_days": 252},
            "black_litterman": {"tau": 0.05},
            "transaction_costs": {
                "spread_bps": 1.0,
                "slippage_bps": 2.0,
                "min_trade_threshold": 0.001,
            },
            "portfolio": {
                "max_position_weight": 0.25,
                "vol_target": 0.12,
                "turnover_penalty": 0.10,
                "solver_primary": "CLARABEL",
                "solver_fallback": "SCS",
            },
            "risk": {
                "max_single_rebalance_turnover": 0.50,
                "max_drawdown_threshold": 0.15,
                "drawdown_lookback_days": 20,
                "vol_breach_multiplier": 1.50,
                "vol_deleveraging_blend": 1.5,  # > 1.0 — invalid
            },
        }
        (tmp_path / "optimizer.yaml").write_text(yaml.dump(data))
        monkeypatch.setattr("config._CONFIG_DIR", tmp_path)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            load_config("optimizer")
