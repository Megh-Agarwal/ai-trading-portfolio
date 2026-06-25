"""Tests for src/agents/pipeline.py — Ticket 2.6."""

from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agents.pipeline import _write_neutral_signals, run_agent_pipeline
from db.models import AgentCall, Base, Signal, View

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATE = datetime.date(2024, 6, 7)
_SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"]

_VALID_NEWS = {
    "sector_sentiments": {s: 0.2 for s in _SECTORS},
    "conviction": 0.7,
    "key_themes": ["tech rally"],
}
_VALID_MACRO = {
    "reasoning": "VIX is low, yield curve steepening.",
    "regime": "risk_on",
    "rate_outlook": "stable",
    "confidence": 0.8,
    "rationale": "Broad risk-on environment.",
}
_VALID_POLY = {
    "implied_prob": {"609655": 0.4},
    "sector_impacts": {s: 0.1 for s in _SECTORS},
    "driving_events": {"XLK": ["Fed rate cut by July 2026?"]},
    "time_horizon": "medium",
    "overall_confidence": 0.65,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _llm_response(payload: dict) -> dict:
    return {
        "id": "msg_test",
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "model": "claude-haiku-4-5-20251001",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def _make_fake_cache(news=None, macro=None, poly=None):
    """Return a cache callable that returns canned responses per agent_name."""
    responses = {
        "sentiment": _llm_response(news or _VALID_NEWS),
        "macro": _llm_response(macro or _VALID_MACRO),
        "events": _llm_response(poly or _VALID_POLY),
    }

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        resp = responses.get(agent_name, _llm_response(news or _VALID_NEWS))
        # Write a minimal agent_call row so cost tracking works.
        row = AgentCall(
            timestamp=datetime.datetime.utcnow(),
            agent_name=agent_name,
            model_string=model,
            prompt_hash="aa",
            input_hash="bb",
            response_json=json.dumps(resp),
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.00012,
            latency_ms=200.0,
            cached=False,
        )
        if engine is not None:
            with Session(engine) as s:
                s.add(row)
                s.commit()
        return resp

    return fake_cache


def _patch_agents(cache_fn):
    """Return context managers that inject `cache_fn` into all three agents."""
    return [
        patch(
            "agents.news_agent.NewsAgent.__init__",
            lambda self, cache=None: super(type(self), self).__init__(
                model_string="claude-haiku-4-5-20251001",
                prompt_template_path=_prompt_path("news_sentiment.txt"),
                cache=cache_fn,
            ),
        ),
    ]


def _prompt_path(name: str):
    from pathlib import Path

    return Path(__file__).parent.parent / "prompts" / name


# ---------------------------------------------------------------------------
# Fixture: patched pipeline where all agents use fake_cache
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_pipeline(tmp_path):
    """Run pipeline with all three agents injected with a fake cache.

    Uses patch on each Agent class's __init__ to inject our fake_cache.
    """
    cache = _make_fake_cache()
    db = _make_engine()

    def patched_news_init(self, cache=None):
        from agents.base import BaseAgent

        BaseAgent.__init__(
            self,
            model_string="claude-haiku-4-5-20251001",
            prompt_template_path=_prompt_path("news_sentiment.txt"),
            cache=cache,
        )

    def patched_macro_init(self, cache=None):
        from agents.base import BaseAgent

        BaseAgent.__init__(
            self,
            model_string="claude-sonnet-4-6",
            prompt_template_path=_prompt_path("macro_regime.txt"),
            cache=cache,
        )

    def patched_poly_init(self, cache=None):
        from agents.base import BaseAgent

        BaseAgent.__init__(
            self,
            model_string="claude-haiku-4-5-20251001",
            prompt_template_path=_prompt_path("polymarket_events.txt"),
            cache=cache,
        )
        self._curated = []

    return cache, db, patched_news_init, patched_macro_init, patched_poly_init


# ---------------------------------------------------------------------------
# Helper: run pipeline with all agents returning valid canned responses
# ---------------------------------------------------------------------------


def _run_pipeline_patched(date, db, cache):
    """Run pipeline with all three agents using `cache` for LLM calls.

    Patches prepare_input on each agent to return minimal valid dicts and
    injects `cache` so no real API calls happen.
    """
    _news_input = {
        "analysis_date": date.isoformat(),
        "week_start": (date - datetime.timedelta(days=7)).isoformat(),
        "sectors": {s: [] for s in _SECTORS},
    }
    _macro_input = {
        "analysis_date": date.isoformat(),
        "series_30d": {},
        "derived_features": {},
        "news_digest": {"XLF": [], "XLI": []},
    }
    _poly_input = {
        "analysis_date": date.isoformat(),
        "markets": [],
    }

    with (
        patch("agents.pipeline.NewsAgent") as MockNews,
        patch("agents.pipeline.MacroAgent") as MockMacro,
        patch("agents.pipeline.PolymarketAgent") as MockPoly,
    ):
        # Configure mock instances
        news_inst = MagicMock()
        news_inst.agent_name = "sentiment"
        news_inst.run.return_value = _VALID_NEWS

        macro_inst = MagicMock()
        macro_inst.agent_name = "macro"
        macro_inst.run.return_value = _VALID_MACRO

        poly_inst = MagicMock()
        poly_inst.agent_name = "events"
        poly_inst.run.return_value = _VALID_POLY

        MockNews.return_value = news_inst
        MockMacro.return_value = macro_inst
        MockPoly.return_value = poly_inst

        # Seed signals that build_views needs (agents are mocked so they won't write)
        with Session(db) as s:
            for sector in _SECTORS:
                s.add(
                    Signal(
                        date=date,
                        agent_name="sentiment",
                        target=sector,
                        signal_value=0.2,
                        confidence=0.7,
                    )
                )
                s.add(
                    Signal(
                        date=date,
                        agent_name="events",
                        target=sector,
                        signal_value=0.1,
                        confidence=0.65,
                    )
                )
            s.add(
                Signal(
                    date=date,
                    agent_name="macro",
                    target="macro_regime",
                    signal_value=1.0,
                    confidence=0.8,
                )
            )
            s.add(
                Signal(
                    date=date,
                    agent_name="macro",
                    target="rate_outlook",
                    signal_value=0.0,
                    confidence=0.8,
                )
            )
            s.commit()

        return run_agent_pipeline(date, db)


# ---------------------------------------------------------------------------
# Core: return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_returns_dict_with_required_keys(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert {
            "date",
            "signals_by_agent",
            "views",
            "total_cost_usd",
            "total_latency_ms",
        } <= result.keys()

    def test_date_is_iso_string(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert result["date"] == _DATE.isoformat()

    def test_signals_by_agent_has_three_keys(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert set(result["signals_by_agent"].keys()) == {"sentiment", "macro", "events"}

    def test_views_has_q_and_omega_diag(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert "q" in result["views"]
        assert "omega_diag" in result["views"]

    def test_views_q_length_equals_n_sectors(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert len(result["views"]["q"]) == len(_SECTORS)

    def test_views_omega_diag_length_equals_n_sectors(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert len(result["views"]["omega_diag"]) == len(_SECTORS)

    def test_all_agents_show_ok_status(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        for name, info in result["signals_by_agent"].items():
            assert info["status"] == "ok", f"{name} status was {info['status']}"

    def test_latency_ms_is_positive(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert result["total_latency_ms"] > 0

    def test_per_agent_latency_present(self):
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        for name, info in result["signals_by_agent"].items():
            assert "latency_ms" in info, f"latency_ms missing for {name}"
            assert info["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_total_cost_is_zero_when_no_new_agent_calls(self):
        # No agent_call rows written (mocked agents don't write them)
        result = _run_pipeline_patched(_DATE, _make_engine(), None)
        assert result["total_cost_usd"] == 0.0

    def test_total_cost_sums_new_agent_call_rows(self):
        db = _make_engine()
        # Pre-existing row (should NOT be counted)
        with Session(db) as s:
            s.add(
                AgentCall(
                    timestamp=datetime.datetime.utcnow(),
                    agent_name="old",
                    model_string="x",
                    prompt_hash="a",
                    input_hash="b",
                    response_json="{}",
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.99,
                    latency_ms=0.0,
                    cached=False,
                )
            )
            s.commit()

        # Seed signals so build_views works
        with Session(db) as s:
            for sector in _SECTORS:
                s.add(
                    Signal(
                        date=_DATE,
                        agent_name="sentiment",
                        target=sector,
                        signal_value=0.0,
                        confidence=0.0,
                    )
                )
                s.add(
                    Signal(
                        date=_DATE,
                        agent_name="events",
                        target=sector,
                        signal_value=0.0,
                        confidence=0.0,
                    )
                )
            s.add(
                Signal(
                    date=_DATE,
                    agent_name="macro",
                    target="macro_regime",
                    signal_value=0.0,
                    confidence=0.0,
                )
            )
            s.add(
                Signal(
                    date=_DATE,
                    agent_name="macro",
                    target="rate_outlook",
                    signal_value=0.0,
                    confidence=0.0,
                )
            )
            s.commit()

        # New rows added AFTER the floor
        with (
            patch("agents.pipeline.NewsAgent") as MockNews,
            patch("agents.pipeline.MacroAgent") as MockMacro,
            patch("agents.pipeline.PolymarketAgent") as MockPoly,
        ):

            def side_effect_run(date, db_engine, portfolio_id="live"):
                with Session(db_engine) as s:
                    s.add(
                        AgentCall(
                            timestamp=datetime.datetime.utcnow(),
                            agent_name="sentiment",
                            model_string="x",
                            prompt_hash="a",
                            input_hash="b",
                            response_json="{}",
                            tokens_in=0,
                            tokens_out=0,
                            cost_usd=0.00012,
                            latency_ms=100.0,
                            cached=False,
                        )
                    )
                    s.commit()
                return _VALID_NEWS

            news_inst = MagicMock()
            news_inst.agent_name = "sentiment"
            news_inst.run.side_effect = side_effect_run
            macro_inst = MagicMock()
            macro_inst.agent_name = "macro"
            macro_inst.run.return_value = _VALID_MACRO
            poly_inst = MagicMock()
            poly_inst.agent_name = "events"
            poly_inst.run.return_value = _VALID_POLY
            MockNews.return_value = news_inst
            MockMacro.return_value = macro_inst
            MockPoly.return_value = poly_inst

            result = run_agent_pipeline(_DATE, db)

        assert abs(result["total_cost_usd"] - 0.00012) < 1e-9
        # Pre-existing $0.99 row must NOT be included
        assert result["total_cost_usd"] < 0.01


# ---------------------------------------------------------------------------
# Failure handling: one agent fails
# ---------------------------------------------------------------------------


class TestPartialFailure:
    def _run_with_one_failing(self, failing_agent: str):
        db = _make_engine()
        # Pre-seed signals for the two agents that succeed
        surviving = [a for a in ["sentiment", "macro", "events"] if a != failing_agent]
        with Session(db) as s:
            for a in surviving:
                if a == "sentiment":
                    for sec in _SECTORS:
                        s.add(
                            Signal(
                                date=_DATE,
                                agent_name="sentiment",
                                target=sec,
                                signal_value=0.2,
                                confidence=0.7,
                            )
                        )
                elif a == "macro":
                    s.add(
                        Signal(
                            date=_DATE,
                            agent_name="macro",
                            target="macro_regime",
                            signal_value=0.0,
                            confidence=0.8,
                        )
                    )
                    s.add(
                        Signal(
                            date=_DATE,
                            agent_name="macro",
                            target="rate_outlook",
                            signal_value=0.0,
                            confidence=0.8,
                        )
                    )
                elif a == "events":
                    for sec in _SECTORS:
                        s.add(
                            Signal(
                                date=_DATE,
                                agent_name="events",
                                target=sec,
                                signal_value=0.0,
                                confidence=0.6,
                            )
                        )
            s.commit()

        with (
            patch("agents.pipeline.NewsAgent") as MockNews,
            patch("agents.pipeline.MacroAgent") as MockMacro,
            patch("agents.pipeline.PolymarketAgent") as MockPoly,
        ):
            news_inst = MagicMock()
            news_inst.agent_name = "sentiment"
            macro_inst = MagicMock()
            macro_inst.agent_name = "macro"
            poly_inst = MagicMock()
            poly_inst.agent_name = "events"

            for inst, name in [
                (news_inst, "sentiment"),
                (macro_inst, "macro"),
                (poly_inst, "events"),
            ]:
                if name == failing_agent:
                    inst.run.side_effect = RuntimeError(f"Simulated {name} failure")
                else:
                    results = {
                        "sentiment": _VALID_NEWS,
                        "macro": _VALID_MACRO,
                        "events": _VALID_POLY,
                    }
                    inst.run.return_value = results[name]

            MockNews.return_value = news_inst
            MockMacro.return_value = macro_inst
            MockPoly.return_value = poly_inst

            return run_agent_pipeline(_DATE, db), db

    def test_pipeline_does_not_crash_when_news_agent_fails(self):
        result, _ = self._run_with_one_failing("sentiment")
        assert result["signals_by_agent"]["sentiment"]["status"] == "error"

    def test_pipeline_does_not_crash_when_macro_agent_fails(self):
        result, _ = self._run_with_one_failing("macro")
        assert result["signals_by_agent"]["macro"]["status"] == "error"

    def test_pipeline_does_not_crash_when_poly_agent_fails(self):
        result, _ = self._run_with_one_failing("events")
        assert result["signals_by_agent"]["events"]["status"] == "error"

    def test_surviving_agents_show_ok_status(self):
        result, _ = self._run_with_one_failing("sentiment")
        assert result["signals_by_agent"]["macro"]["status"] == "ok"
        assert result["signals_by_agent"]["events"]["status"] == "ok"

    def test_error_message_captured_in_result(self):
        result, _ = self._run_with_one_failing("macro")
        assert "Simulated macro failure" in result["signals_by_agent"]["macro"]["error"]

    def test_neutral_stubs_written_for_failed_sentiment_agent(self):
        _, db = self._run_with_one_failing("sentiment")
        with Session(db) as s:
            rows = (
                s.execute(
                    select(Signal)
                    .where(Signal.date == _DATE)
                    .where(Signal.agent_name == "sentiment")
                )
                .scalars()
                .all()
            )
        assert len(rows) == len(_SECTORS)
        assert all(r.signal_value == 0.0 for r in rows)
        assert all(r.confidence == 0.0 for r in rows)

    def test_neutral_stubs_written_for_failed_macro_agent(self):
        _, db = self._run_with_one_failing("macro")
        with Session(db) as s:
            rows = (
                s.execute(
                    select(Signal).where(Signal.date == _DATE).where(Signal.agent_name == "macro")
                )
                .scalars()
                .all()
            )
        targets = {r.target for r in rows}
        assert "macro_regime" in targets
        assert "rate_outlook" in targets

    def test_views_still_written_when_one_agent_fails(self):
        _, db = self._run_with_one_failing("events")
        with Session(db) as s:
            rows = s.execute(select(View).where(View.date == _DATE)).scalars().all()
        assert len(rows) == len(_SECTORS)

    def test_partial_failure_returns_zero_q_for_failed_sector_signal(self):
        """When sentiment fails (zero stubs) and poly succeeds with 0, Q should be small."""
        result, _ = self._run_with_one_failing("sentiment")
        # All Q values should be finite (no NaN/inf)
        assert all(abs(v) < 1.0 for v in result["views"]["q"])


# ---------------------------------------------------------------------------
# All agents fail → RuntimeError
# ---------------------------------------------------------------------------


class TestAllAgentsFail:
    def test_raises_runtime_error_when_all_agents_fail(self):
        db = _make_engine()
        with (
            patch("agents.pipeline.NewsAgent") as MockNews,
            patch("agents.pipeline.MacroAgent") as MockMacro,
            patch("agents.pipeline.PolymarketAgent") as MockPoly,
        ):
            for Mock, name in [(MockNews, "sentiment"), (MockMacro, "macro"), (MockPoly, "events")]:
                inst = MagicMock()
                inst.agent_name = name
                inst.run.side_effect = RuntimeError(f"{name} boom")
                Mock.return_value = inst

            with pytest.raises(RuntimeError, match="All three agents failed"):
                run_agent_pipeline(_DATE, db)

    def test_all_fail_error_includes_date(self):
        db = _make_engine()
        with (
            patch("agents.pipeline.NewsAgent") as MockNews,
            patch("agents.pipeline.MacroAgent") as MockMacro,
            patch("agents.pipeline.PolymarketAgent") as MockPoly,
        ):
            for Mock, name in [(MockNews, "sentiment"), (MockMacro, "macro"), (MockPoly, "events")]:
                inst = MagicMock()
                inst.agent_name = name
                inst.run.side_effect = RuntimeError("boom")
                Mock.return_value = inst

            with pytest.raises(RuntimeError) as exc_info:
                run_agent_pipeline(_DATE, db)

        assert str(_DATE) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Neutral stub helper
# ---------------------------------------------------------------------------


class TestWriteNeutralSignals:
    def test_writes_10_rows_for_sentiment(self):
        db = _make_engine()
        _write_neutral_signals("sentiment", _DATE, db)
        with Session(db) as s:
            rows = s.execute(select(Signal).where(Signal.agent_name == "sentiment")).scalars().all()
        assert len(rows) == len(_SECTORS)

    def test_writes_2_rows_for_macro(self):
        db = _make_engine()
        _write_neutral_signals("macro", _DATE, db)
        with Session(db) as s:
            rows = s.execute(select(Signal).where(Signal.agent_name == "macro")).scalars().all()
        assert len(rows) == 2
        targets = {r.target for r in rows}
        assert targets == {"macro_regime", "rate_outlook"}

    def test_writes_10_rows_for_events(self):
        db = _make_engine()
        _write_neutral_signals("events", _DATE, db)
        with Session(db) as s:
            rows = s.execute(select(Signal).where(Signal.agent_name == "events")).scalars().all()
        assert len(rows) == len(_SECTORS)

    def test_stubs_are_idempotent(self):
        db = _make_engine()
        _write_neutral_signals("sentiment", _DATE, db)
        _write_neutral_signals("sentiment", _DATE, db)
        with Session(db) as s:
            count = len(
                s.execute(select(Signal).where(Signal.agent_name == "sentiment")).scalars().all()
            )
        assert count == len(_SECTORS)

    def test_stubs_all_have_zero_signal_and_confidence(self):
        db = _make_engine()
        _write_neutral_signals("events", _DATE, db)
        with Session(db) as s:
            rows = s.execute(select(Signal)).scalars().all()
        assert all(r.signal_value == 0.0 for r in rows)
        assert all(r.confidence == 0.0 for r in rows)

    def test_unknown_agent_name_is_handled_gracefully(self):
        db = _make_engine()
        _write_neutral_signals("nonexistent", _DATE, db)  # should not raise
        with Session(db) as s:
            count = len(s.execute(select(Signal)).scalars().all())
        assert count == 0
