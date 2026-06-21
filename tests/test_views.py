"""Tests for src/aggregator/views.py — Ticket 2.5."""
from __future__ import annotations

import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from aggregator.views import _MIN_CONVICTION, build_views
from db.models import Base, Signal, View

# These constants were moved to config/optimizer.yaml (ADR-012 / Blocker 2).
# Test file keeps local copies matching the config defaults so arithmetic tests pass.
_MAX_EXCESS_RETURN_ANNUAL: float = 0.05
_OMEGA_BASE: float = 0.0001
_WEEKS_PER_YEAR: int = 52

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = datetime.date(2024, 1, 5)
_SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"]
_N = len(_SECTORS)


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed_signals(
    db,
    *,
    date: datetime.date = _DATE,
    news_signal: float = 0.0,
    news_conv: float = 0.0,
    poly_signal: float = 0.0,
    poly_conf: float = 0.0,
    macro_regime: float = 0.0,   # neutral
    macro_conf: float = 0.0,
):
    """Seed uniform signals across all sectors for test simplicity."""
    rows = []
    for sector in _SECTORS:
        rows.append(
            Signal(
                date=date,
                agent_name="sentiment",
                target=sector,
                signal_value=news_signal,
                confidence=news_conv,
            )
        )
        rows.append(
            Signal(
                date=date,
                agent_name="events",
                target=sector,
                signal_value=poly_signal,
                confidence=poly_conf,
            )
        )
    rows.append(
        Signal(
            date=date,
            agent_name="macro",
            target="macro_regime",
            signal_value=macro_regime,
            confidence=macro_conf,
        )
    )
    rows.append(
        Signal(
            date=date,
            agent_name="macro",
            target="rate_outlook",
            signal_value=0.0,
            confidence=macro_conf,
        )
    )
    with Session(db) as session:
        session.add_all(rows)
        session.commit()


# ---------------------------------------------------------------------------
# Return shape and type
# ---------------------------------------------------------------------------


class TestReturnShape:
    def test_q_is_1d_array_of_length_n(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.5, news_conv=0.8)
        q, _ = build_views(_DATE, db)
        assert isinstance(q, np.ndarray)
        assert q.shape == (_N,)

    def test_omega_is_n_by_n_diagonal(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.5, news_conv=0.8)
        _, omega = build_views(_DATE, db)
        assert omega.shape == (_N, _N)
        # Off-diagonal must be zero
        np.testing.assert_array_equal(omega - np.diag(np.diag(omega)), 0)

    def test_q_dtype_is_float(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.3, news_conv=0.5)
        q, _ = build_views(_DATE, db)
        assert q.dtype == float


# ---------------------------------------------------------------------------
# Edge case: zero signals → zero Q, max Omega uncertainty
# ---------------------------------------------------------------------------


class TestZeroSignal:
    def test_q_all_zero_when_all_signals_are_zero(self):
        db = _make_engine()
        _seed_signals(db)  # all defaults = 0
        q, _ = build_views(_DATE, db)
        np.testing.assert_array_equal(q, 0.0)

    def test_omega_at_max_uncertainty_when_conviction_is_zero(self):
        db = _make_engine()
        _seed_signals(db)
        _, omega = build_views(_DATE, db)
        expected = _OMEGA_BASE * _WEEKS_PER_YEAR / _MIN_CONVICTION
        np.testing.assert_allclose(np.diag(omega), expected, rtol=1e-9)

    def test_views_written_to_db_with_zero_expected_return(self):
        db = _make_engine()
        _seed_signals(db)
        build_views(_DATE, db)
        with Session(db) as s:
            rows = s.execute(select(View).where(View.date == _DATE)).scalars().all()
        assert len(rows) == _N
        assert all(r.expected_return == 0.0 for r in rows)


# ---------------------------------------------------------------------------
# Edge case: macro risk_off + bullish news → dampened view
# ---------------------------------------------------------------------------


class TestMacroRiskOffDampening:
    """Explicit test from ticket acceptance criteria."""

    def _q_no_regime(self, news_signal, news_conv, poly_signal=0.0, poly_conf=0.0):
        """Expected Q value with neutral regime (scale=0.75) for comparison."""
        weights = {"news": 0.4, "macro": 0.3, "polymarket": 0.3}
        raw = weights["news"] * news_signal * news_conv + weights["polymarket"] * poly_signal * poly_conf
        return raw * 0.75 * _MAX_EXCESS_RETURN_ANNUAL   # neutral scale = 0.75

    def test_risk_off_produces_smaller_q_than_neutral(self):
        db_risk_off = _make_engine()
        _seed_signals(db_risk_off, news_signal=0.8, news_conv=0.9, macro_regime=-1.0)
        q_riskoff, _ = build_views(_DATE, db_risk_off)

        db_neutral = _make_engine()
        _seed_signals(db_neutral, news_signal=0.8, news_conv=0.9, macro_regime=0.0)
        q_neutral, _ = build_views(_DATE, db_neutral)

        # All sectors are uniform; risk_off must produce strictly smaller |Q|
        assert np.all(np.abs(q_riskoff) < np.abs(q_neutral))

    def test_risk_off_produces_smaller_q_than_risk_on(self):
        db_risk_off = _make_engine()
        _seed_signals(db_risk_off, news_signal=0.8, news_conv=0.9, macro_regime=-1.0)
        q_riskoff, _ = build_views(_DATE, db_risk_off)

        db_risk_on = _make_engine()
        _seed_signals(db_risk_on, news_signal=0.8, news_conv=0.9, macro_regime=1.0)
        q_riskon, _ = build_views(_DATE, db_risk_on)

        assert np.all(np.abs(q_riskoff) < np.abs(q_riskon))

    def test_risk_off_dampens_q_by_half_vs_risk_on(self):
        """risk_off scale=0.50, risk_on scale=1.00 → Q ratio = 0.50."""
        db_off = _make_engine()
        _seed_signals(db_off, news_signal=0.6, news_conv=1.0, macro_regime=-1.0)
        q_off, _ = build_views(_DATE, db_off)

        db_on = _make_engine()
        _seed_signals(db_on, news_signal=0.6, news_conv=1.0, macro_regime=1.0)
        q_on, _ = build_views(_DATE, db_on)

        # ratio should be 0.50 / 1.00 = 0.5 everywhere (avoid div-by-zero via nonzero news)
        ratios = q_off / q_on
        np.testing.assert_allclose(ratios, 0.5, rtol=1e-9)

    def test_risk_off_omega_larger_than_risk_on(self):
        """Risk-off reduces conviction → larger Omega entries (more uncertainty)."""
        db_off = _make_engine()
        _seed_signals(db_off, news_signal=0.6, news_conv=0.8, macro_conf=0.7, macro_regime=-1.0)
        _, omega_off = build_views(_DATE, db_off)

        db_on = _make_engine()
        _seed_signals(db_on, news_signal=0.6, news_conv=0.8, macro_conf=0.7, macro_regime=1.0)
        _, omega_on = build_views(_DATE, db_on)

        # Higher Omega entries = lower conviction in risk-off
        assert np.all(np.diag(omega_off) > np.diag(omega_on))


# ---------------------------------------------------------------------------
# Q magnitude arithmetic
# ---------------------------------------------------------------------------


class TestQArithmetic:
    def test_unit_signal_unit_conviction_neutral_regime_q_value(self):
        """news_signal=1, news_conv=1, neutral macro (scale=0.75), poly=0.
        Expected Q = 0.4 * 1 * 1 * 0.75 * (0.05/52).
        """
        db = _make_engine()
        _seed_signals(db, news_signal=1.0, news_conv=1.0, macro_regime=0.0)
        q, _ = build_views(_DATE, db)
        weights_news = 0.4
        expected = weights_news * 1.0 * 1.0 * 0.75 * _MAX_EXCESS_RETURN_ANNUAL
        np.testing.assert_allclose(q, expected, rtol=1e-9)

    def test_q_is_positive_for_positive_signal(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.5, news_conv=0.8)
        q, _ = build_views(_DATE, db)
        assert np.all(q > 0)

    def test_q_is_negative_for_negative_signal(self):
        db = _make_engine()
        _seed_signals(db, news_signal=-0.5, news_conv=0.8)
        q, _ = build_views(_DATE, db)
        assert np.all(q < 0)

    def test_q_respects_agent_weights(self):
        """Doubling news weight vs polymarket weight changes Q proportionally."""
        weights_heavy_news = {"news": 0.6, "macro": 0.3, "polymarket": 0.1}
        weights_heavy_poly = {"news": 0.1, "macro": 0.3, "polymarket": 0.6}

        db1 = _make_engine()
        _seed_signals(db1, news_signal=1.0, news_conv=1.0, poly_signal=0.0)
        q1, _ = build_views(_DATE, db1, weights=weights_heavy_news)

        db2 = _make_engine()
        _seed_signals(db2, news_signal=0.0, poly_signal=1.0, poly_conf=1.0)
        q2, _ = build_views(_DATE, db2, weights=weights_heavy_poly)

        # Both should be positive; heavy-news run uses w=0.6, heavy-poly uses w=0.6
        np.testing.assert_allclose(q1, q2, rtol=1e-9)


# ---------------------------------------------------------------------------
# Omega arithmetic
# ---------------------------------------------------------------------------


class TestOmegaArithmetic:
    def test_higher_conviction_produces_smaller_omega(self):
        db_low = _make_engine()
        _seed_signals(db_low, news_signal=0.5, news_conv=0.1)
        _, omega_low = build_views(_DATE, db_low)

        db_high = _make_engine()
        _seed_signals(db_high, news_signal=0.5, news_conv=0.9)
        _, omega_high = build_views(_DATE, db_high)

        assert np.all(np.diag(omega_low) > np.diag(omega_high))

    def test_omega_inversely_proportional_to_conviction(self):
        """At zero poly/macro: omega = OMEGA_BASE / (w_news * news_conv * regime_scale)."""
        news_conv = 0.8
        macro_regime = 0.0   # scale = 0.75
        regime_scale = 0.75
        w_news = 0.4
        # macro_conf=0 and poly_conf=0, so agg_conviction = w_news * news_conv * scale
        expected_conviction = w_news * news_conv * regime_scale
        expected_omega = _OMEGA_BASE * _WEEKS_PER_YEAR / max(expected_conviction, _MIN_CONVICTION)

        db = _make_engine()
        _seed_signals(db, news_signal=0.5, news_conv=news_conv)
        _, omega = build_views(_DATE, db)
        np.testing.assert_allclose(np.diag(omega), expected_omega, rtol=1e-9)


# ---------------------------------------------------------------------------
# DB persistence and idempotency
# ---------------------------------------------------------------------------


class TestDBPersistence:
    def test_views_written_to_db(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.4, news_conv=0.7)
        build_views(_DATE, db)

        with Session(db) as s:
            rows = s.execute(select(View).where(View.date == _DATE)).scalars().all()
        assert len(rows) == _N

    def test_views_db_rows_have_correct_sectors(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.4, news_conv=0.7)
        build_views(_DATE, db)

        with Session(db) as s:
            rows = s.execute(select(View).where(View.date == _DATE)).scalars().all()
        stored_sectors = {r.sector for r in rows}
        assert stored_sectors == set(_SECTORS)

    def test_views_idempotent_on_rerun(self):
        """Re-running build_views for the same date should not duplicate rows."""
        db = _make_engine()
        _seed_signals(db, news_signal=0.4, news_conv=0.7)
        build_views(_DATE, db)
        build_views(_DATE, db)  # second run

        with Session(db) as s:
            rows = s.execute(select(View).where(View.date == _DATE)).scalars().all()
        assert len(rows) == _N  # still exactly N, not 2*N

    def test_views_reconstructable_for_historical_date(self):
        """Views written for an old date are still readable later."""
        old_date = datetime.date(2023, 6, 9)
        db = _make_engine()
        _seed_signals(db, date=old_date, news_signal=0.3, news_conv=0.6)
        q_original, _ = build_views(old_date, db)

        with Session(db) as s:
            rows = (
                s.execute(select(View).where(View.date == old_date))
                .scalars()
                .all()
            )
        stored_q = {r.sector: r.expected_return for r in rows}
        np.testing.assert_allclose(
            [stored_q[sec] for sec in _SECTORS], q_original, rtol=1e-9
        )

    def test_views_for_two_dates_are_independent(self):
        date1 = datetime.date(2024, 1, 5)
        date2 = datetime.date(2024, 1, 12)
        db = _make_engine()
        _seed_signals(db, date=date1, news_signal=0.5, news_conv=0.8)
        _seed_signals(db, date=date2, news_signal=-0.3, news_conv=0.6)
        build_views(date1, db)
        build_views(date2, db)

        with Session(db) as s:
            count = len(s.execute(select(View)).scalars().all())
        assert count == 2 * _N


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_raises_if_no_signals_in_db(self):
        db = _make_engine()
        with pytest.raises(ValueError, match="No signal rows found"):
            build_views(_DATE, db)

    def test_default_weights_used_when_none_passed(self):
        """Passing weights=None should produce same result as explicit defaults."""
        db1 = _make_engine()
        _seed_signals(db1, news_signal=0.5, news_conv=0.8)
        q_default, _ = build_views(_DATE, db1, weights=None)

        db2 = _make_engine()
        _seed_signals(db2, news_signal=0.5, news_conv=0.8)
        q_explicit, _ = build_views(_DATE, db2, weights={"news": 0.4, "macro": 0.3, "polymarket": 0.3})

        np.testing.assert_array_equal(q_default, q_explicit)

    def test_missing_agent_data_treated_as_zero(self):
        """If one agent has no rows, that agent contributes 0 to the signal."""
        db = _make_engine()
        # Seed only the macro signal — no sentiment or events rows
        with Session(db) as s:
            s.add(Signal(date=_DATE, agent_name="macro", target="macro_regime",
                         signal_value=0.0, confidence=0.8))
            s.commit()
        q, _ = build_views(_DATE, db)
        np.testing.assert_array_equal(q, 0.0)

    def test_raises_if_weights_sum_exceeds_one(self):
        db = _make_engine()
        _seed_signals(db, news_signal=0.5, news_conv=0.8)
        with pytest.raises(ValueError, match="weights must sum to"):
            build_views(_DATE, db, weights={"news": 0.5, "macro": 0.5, "polymarket": 0.5})

    def test_backtest_mode_ignores_polymarket_signal(self):
        """In backtest mode polymarket weight=0.0 — poly_signal has no effect on Q."""
        db = _make_engine()
        _seed_signals(db, poly_signal=1.0, poly_conf=1.0)
        q, _ = build_views(_DATE, db, mode="backtest")
        np.testing.assert_array_equal(q, 0.0)
