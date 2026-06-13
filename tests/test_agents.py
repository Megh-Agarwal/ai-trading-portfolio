"""Smoke tests for BaseAgent, the three agent output schemas, and NewsAgent."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from agents.base import BaseAgent
from agents.schemas import MacroRegimeSignal, NewsSignal, PolymarketSignal
from db.models import Base, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _llm_response(payload: dict) -> dict:
    """Wrap a dict as a minimal Anthropic-shaped response."""
    return {
        "id": "msg_test",
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "model": "claude-haiku-4-5-20251001",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing BaseAgent mechanics
# ---------------------------------------------------------------------------


class _FakeAgent(BaseAgent):
    agent_name = "fake"
    _schema_class = NewsSignal

    def prepare_input(self, date: datetime.date, db) -> dict:
        return {"date": str(date)}

    def _write_signals(self, date, validated, call_id, db) -> None:
        from sqlalchemy.orm import Session
        from db.models import Signal

        rows = [
            Signal(
                date=date,
                agent_name=self.agent_name,
                target=sector,
                signal_value=sentiment,
                confidence=validated["conviction"],
                raw_call_id=call_id,
            )
            for sector, sentiment in validated["sector_sentiments"].items()
        ]
        self._insert_signals(rows, db)


@pytest.fixture()
def prompt_file(tmp_path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text("You are a trading agent. Return JSON.")
    return p


# ---------------------------------------------------------------------------
# BaseAgent — abstract enforcement
# ---------------------------------------------------------------------------


def test_base_agent_cannot_be_instantiated_directly(prompt_file):
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseAgent("claude-haiku-4-5-20251001", prompt_file)  # type: ignore[abstract]


def test_base_agent_raises_on_missing_prompt():
    with pytest.raises(FileNotFoundError):
        _FakeAgent("claude-haiku-4-5-20251001", "/nonexistent/prompt.txt")


# ---------------------------------------------------------------------------
# NewsSignal schema
# ---------------------------------------------------------------------------

_VALID_NEWS = {
    "sector_sentiments": {"XLK": 0.8, "XLF": -0.3, "XLV": 0.0},
    "conviction": 0.75,
    "key_themes": ["rate hike fears", "AI spending boom"],
}


def test_news_signal_valid():
    sig = NewsSignal.model_validate(_VALID_NEWS)
    assert sig.sector_sentiments["XLK"] == 0.8
    assert sig.conviction == 0.75
    assert len(sig.key_themes) == 2


def test_news_signal_sentiment_out_of_range():
    bad = {**_VALID_NEWS, "sector_sentiments": {"XLK": 1.5}}
    with pytest.raises(ValidationError, match="out of"):
        NewsSignal.model_validate(bad)


def test_news_signal_conviction_out_of_range():
    bad = {**_VALID_NEWS, "conviction": 1.1}
    with pytest.raises(ValidationError):
        NewsSignal.model_validate(bad)


def test_news_signal_missing_field():
    with pytest.raises(ValidationError):
        NewsSignal.model_validate({"conviction": 0.5, "key_themes": []})


# ---------------------------------------------------------------------------
# MacroRegimeSignal schema
# ---------------------------------------------------------------------------

_VALID_MACRO = {
    "reasoning": "VIX elevated and rising. Yield curve deeply inverted. FOMC hawkish language persists.",
    "regime": "risk_off",
    "rate_outlook": "rising",
    "confidence": 0.9,
    "rationale": "Yield curve inversion deepening with FOMC hawkish tone.",
}


def test_macro_signal_valid():
    sig = MacroRegimeSignal.model_validate(_VALID_MACRO)
    assert sig.regime == "risk_off"
    assert sig.rate_outlook == "rising"
    assert sig.confidence == 0.9


def test_macro_signal_invalid_regime():
    bad = {**_VALID_MACRO, "regime": "super_bullish"}
    with pytest.raises(ValidationError):
        MacroRegimeSignal.model_validate(bad)


def test_macro_signal_invalid_rate_outlook():
    bad = {**_VALID_MACRO, "rate_outlook": "sideways"}
    with pytest.raises(ValidationError):
        MacroRegimeSignal.model_validate(bad)


def test_macro_signal_confidence_out_of_range():
    bad = {**_VALID_MACRO, "confidence": -0.1}
    with pytest.raises(ValidationError):
        MacroRegimeSignal.model_validate(bad)


# ---------------------------------------------------------------------------
# PolymarketSignal schema
# ---------------------------------------------------------------------------

_VALID_POLY = {
    "implied_prob": {"609655": 0.72, "612300": 0.35},
    "sector_impacts": {"XLF": 0.5, "XLU": -0.3},
    "time_horizon": "short",
}


def test_polymarket_signal_valid():
    sig = PolymarketSignal.model_validate(_VALID_POLY)
    assert sig.implied_prob["609655"] == 0.72
    assert sig.sector_impacts["XLF"] == 0.5


def test_polymarket_signal_prob_out_of_range():
    bad = {**_VALID_POLY, "implied_prob": {"609655": 1.05}}
    with pytest.raises(ValidationError, match="out of"):
        PolymarketSignal.model_validate(bad)


def test_polymarket_signal_impact_out_of_range():
    bad = {**_VALID_POLY, "sector_impacts": {"XLF": -1.5}}
    with pytest.raises(ValidationError, match="out of"):
        PolymarketSignal.model_validate(bad)


def test_polymarket_signal_missing_time_horizon():
    bad = {"implied_prob": {"609655": 0.5}, "sector_impacts": {"XLF": 0.1}}
    with pytest.raises(ValidationError):
        PolymarketSignal.model_validate(bad)


# ---------------------------------------------------------------------------
# BaseAgent.validate_output
# ---------------------------------------------------------------------------


def test_validate_output_parses_valid_response(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    result = agent.validate_output(_llm_response(_VALID_NEWS))
    assert result["conviction"] == 0.75
    assert result["sector_sentiments"]["XLK"] == 0.8


def test_validate_output_strips_markdown_fences(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    wrapped = {
        "content": [{"type": "text", "text": f"```json\n{json.dumps(_VALID_NEWS)}\n```"}],
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }
    result = agent.validate_output(wrapped)
    assert result["conviction"] == 0.75


def test_validate_output_raises_on_invalid_json(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    bad_response = {"content": [{"type": "text", "text": "not json at all"}], "usage": {}}
    with pytest.raises(ValueError, match="not valid JSON"):
        agent.validate_output(bad_response)


def test_validate_output_raises_on_schema_violation(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    bad_payload = {"sector_sentiments": {"XLK": 99.0}, "conviction": 0.5, "key_themes": []}
    with pytest.raises(ValueError):
        agent.validate_output(_llm_response(bad_payload))


def test_validate_output_raises_on_empty_content(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    with pytest.raises(ValueError, match="no text content"):
        agent.validate_output({"content": [], "usage": {}})


# ---------------------------------------------------------------------------
# BaseAgent.run — end-to-end with mocked cached_call
# ---------------------------------------------------------------------------


def test_run_writes_signals_to_db(prompt_file):
    db = _make_engine()

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        return _llm_response(_VALID_NEWS)

    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file, cache=fake_cache)
    result = agent.run(datetime.date(2024, 1, 15), db)

    assert result["conviction"] == 0.75

    from sqlalchemy.orm import Session
    with Session(db) as s:
        rows = s.query(Signal).all()

    assert len(rows) == 3
    targets = {r.target for r in rows}
    assert targets == {"XLK", "XLF", "XLV"}
    xlk_row = next(r for r in rows if r.target == "XLK")
    assert xlk_row.signal_value == pytest.approx(0.8)
    assert xlk_row.agent_name == "fake"


def test_run_returns_validated_dict(prompt_file):
    db = _make_engine()

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        return _llm_response(_VALID_NEWS)

    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file, cache=fake_cache)
    result = agent.run(datetime.date(2024, 1, 15), db)

    assert set(result.keys()) == {"sector_sentiments", "conviction", "key_themes"}


# ---------------------------------------------------------------------------
# NewsAgent
# ---------------------------------------------------------------------------

_ALL_SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"]

_FULL_NEWS_SIGNAL = {
    "sector_sentiments": {s: 0.0 for s in _ALL_SECTORS} | {"XLK": 0.6, "XLE": -0.4},
    "conviction": 0.55,
    "key_themes": ["AI spending accelerating", "oil supply surplus emerging", "rate cut expectations easing"],
}


def _seed_news(db, sectors: list[str], date: datetime.date) -> None:
    """Insert minimal news_raw rows so prepare_input returns non-empty results."""
    from db.models import NewsRaw

    ts = datetime.datetime.combine(date - datetime.timedelta(days=2), datetime.time(12, 0))
    with Session(db) as s:
        for sector in sectors:
            s.add(NewsRaw(ticker="AAA", sector=sector, timestamp=ts, title=f"{sector} headline"))
        s.commit()


def test_news_agent_uses_haiku_model():
    from agents.news_agent import NewsAgent

    agent = NewsAgent()
    assert "haiku" in agent._model.lower()


def test_news_agent_prepare_input_returns_all_sectors(tmp_path):
    from agents.news_agent import NewsAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_news(db, _ALL_SECTORS[:3], date)

    agent = NewsAgent()
    data = agent.prepare_input(date, db)

    assert data["analysis_date"] == "2024-06-07"
    assert set(data["sectors"].keys()) == set(_ALL_SECTORS)
    # seeded sectors have articles; others are empty lists
    assert len(data["sectors"]["XLK"]) == 1
    assert data["sectors"]["XLU"] == []


def test_news_agent_prepare_input_ignores_articles_outside_window():
    from agents.news_agent import NewsAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    # Insert article 10 days before analysis_date — outside the 7-day window
    old_ts = datetime.datetime.combine(date - datetime.timedelta(days=10), datetime.time(12, 0))
    with Session(db) as s:
        from db.models import NewsRaw
        s.add(NewsRaw(ticker="AAA", sector="XLK", timestamp=old_ts, title="Old headline"))
        s.commit()

    agent = NewsAgent()
    data = agent.prepare_input(date, db)
    assert data["sectors"]["XLK"] == []


def test_news_agent_run_writes_10_signal_rows():
    from agents.news_agent import NewsAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_news(db, _ALL_SECTORS, date)

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        return _llm_response(_FULL_NEWS_SIGNAL)

    agent = NewsAgent(cache=fake_cache)
    result = agent.run(date, db)

    assert result["conviction"] == pytest.approx(0.55)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="sentiment", date=date).all()

    assert len(rows) == 10
    targets = {r.target for r in rows}
    assert targets == set(_ALL_SECTORS)
    xlk = next(r for r in rows if r.target == "XLK")
    assert xlk.signal_value == pytest.approx(0.6)
    assert xlk.confidence == pytest.approx(0.55)


def test_news_agent_run_is_idempotent():
    """Re-running the agent for the same date replaces prior signals, not duplicates."""
    from agents.news_agent import NewsAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_news(db, _ALL_SECTORS, date)

    call_count = 0

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        nonlocal call_count
        call_count += 1
        return _llm_response(_FULL_NEWS_SIGNAL)

    agent = NewsAgent(cache=fake_cache)
    agent.run(date, db)
    agent.run(date, db)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="sentiment", date=date).all()

    # Still exactly 10 rows, not 20
    assert len(rows) == 10


# ---------------------------------------------------------------------------
# MacroRegimeSignal schema
# ---------------------------------------------------------------------------

_VALID_MACRO_SIGNAL = {
    "reasoning": "VIX at 18.5, below 90d avg of 20.1 — mild risk-on. T10Y2Y at -0.3, improving from -0.5 avg. DGS10 fell 25bp in 30d signalling rate relief. CPI YoY 3.2%, declining trend. ICSA stable near 210k. Overall: cautious risk-on.",
    "regime": "neutral",
    "rate_outlook": "falling",
    "confidence": 0.68,
    "rationale": "Mixed but slightly improving macro backdrop. VIX easing and yield curve less inverted point toward neutral-to-risk-on, while still-elevated CPI prevents a full risk-on call. Rate outlook leans falling as the Fed approaches its terminal rate.",
}


def test_macro_regime_signal_valid_with_reasoning():
    sig = MacroRegimeSignal.model_validate(_VALID_MACRO_SIGNAL)
    assert sig.regime == "neutral"
    assert sig.rate_outlook == "falling"
    assert sig.confidence == pytest.approx(0.68)
    assert len(sig.reasoning) > 10
    assert len(sig.rationale) > 10


def test_macro_regime_signal_missing_reasoning():
    bad = {k: v for k, v in _VALID_MACRO_SIGNAL.items() if k != "reasoning"}
    with pytest.raises(ValidationError):
        MacroRegimeSignal.model_validate(bad)


# ---------------------------------------------------------------------------
# MacroAgent
# ---------------------------------------------------------------------------


def _seed_macro(db, date: datetime.date) -> None:
    """Insert minimal macro rows covering last 90 days so prepare_input doesn't return empty."""
    from db.models import Macro

    series_values = {
        "VIXCLS": 18.5, "T10Y2Y": -0.35, "DGS10": 4.30,
        "DTWEXBGS": 103.5, "CPIAUCSL": 309.0, "UNRATE": 3.8, "ICSA": 210000,
    }

    rows = []
    for days_back in range(0, 95, 1):
        d = date - datetime.timedelta(days=days_back)
        for sid, val in series_values.items():
            rows.append(Macro(date=d, series_id=sid, value=val))

    # Seed CPI 12 months back for YoY calculation
    for days_back in range(360, 400):
        d = date - datetime.timedelta(days=days_back)
        rows.append(Macro(date=d, series_id="CPIAUCSL", value=295.0))

    with Session(db) as s:
        s.add_all(rows)
        s.commit()


def test_macro_agent_uses_sonnet_model():
    from agents.macro_agent import MacroAgent

    agent = MacroAgent()
    assert "sonnet" in agent._model.lower()


def test_macro_agent_prepare_input_returns_derived_features():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_macro(db, date)

    agent = MacroAgent()
    data = agent.prepare_input(date, db)

    assert data["analysis_date"] == "2024-06-07"
    features = data["derived_features"]
    assert "vix_current" in features
    assert "t10y2y_current" in features
    assert "dgs10_current" in features
    assert "cpi_yoy_pct" in features
    assert "unrate_current" in features
    assert "icsa_current" in features


def test_macro_agent_prepare_input_series_30d_keys():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_macro(db, date)

    agent = MacroAgent()
    data = agent.prepare_input(date, db)

    # All 7 FRED series should be present
    assert set(data["series_30d"].keys()) == {
        "VIXCLS", "T10Y2Y", "DGS10", "DTWEXBGS", "CPIAUCSL", "UNRATE", "ICSA"
    }
    # 30 days of daily data
    assert len(data["series_30d"]["VIXCLS"]) == 31  # 0..30 inclusive


def test_macro_agent_prepare_input_empty_db_returns_empty_features():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    agent = MacroAgent()
    data = agent.prepare_input(datetime.date(2024, 6, 7), db)

    assert data["derived_features"] == {}
    assert data["series_30d"] == {}


def test_macro_agent_run_writes_2_signal_rows():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_macro(db, date)

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        return _llm_response(_VALID_MACRO_SIGNAL)

    agent = MacroAgent(cache=fake_cache)
    result = agent.run(date, db)

    assert result["regime"] == "neutral"
    assert result["rate_outlook"] == "falling"

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="macro", date=date).all()

    assert len(rows) == 2
    targets = {r.target for r in rows}
    assert targets == {"macro_regime", "rate_outlook"}

    regime_row = next(r for r in rows if r.target == "macro_regime")
    assert regime_row.signal_value == pytest.approx(0.0)  # neutral → 0.0

    rate_row = next(r for r in rows if r.target == "rate_outlook")
    assert rate_row.signal_value == pytest.approx(-1.0)  # falling → -1.0


def test_macro_agent_run_is_idempotent():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_macro(db, date)

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine):
        return _llm_response(_VALID_MACRO_SIGNAL)

    agent = MacroAgent(cache=fake_cache)
    agent.run(date, db)
    agent.run(date, db)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="macro", date=date).all()

    # 2 rows, not 4
    assert len(rows) == 2
