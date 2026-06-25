"""Smoke tests for BaseAgent, the three agent output schemas, and NewsAgent."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

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


def _tool_response(payload: dict, tool_name: str = "fake_tool") -> dict:
    """Wrap a dict as a minimal Anthropic tool-use response."""
    return {
        "id": "msg_test",
        "content": [{"type": "tool_use", "id": "call_123", "name": tool_name, "input": payload}],
        "model": "claude-haiku-4-5-20251001",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing BaseAgent mechanics
# ---------------------------------------------------------------------------


class _FakeAgent(BaseAgent):
    agent_name = "fake"
    _schema_class = NewsSignal
    _tool = {
        "name": "fake_tool",
        "description": "test tool",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }

    def prepare_input(self, date: datetime.date, db) -> dict:
        return {"date": str(date)}

    def _write_signals(self, date, validated, call_id, db, portfolio_id="live") -> None:
        rows = [
            Signal(
                portfolio_id=portfolio_id,
                date=date,
                agent_name=self.agent_name,
                target=sector,
                signal_value=sentiment,
                confidence=validated["sector_conviction"].get(sector, 0.0),
                raw_call_id=call_id,
            )
            for sector, sentiment in validated["sector_sentiments"].items()
        ]
        self._insert_signals(rows, db)


@pytest.fixture()
def prompt_file(tmp_path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text("You are a trading agent.")
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
    "sector_conviction": {"XLK": 0.75, "XLF": 0.6, "XLV": 0.0},
    "key_themes": ["rate hike fears", "AI spending boom", "energy volatility"],
    "evidence": [
        {"sector": "XLK", "headline": "Nvidia beats expectations", "impact": "positive for tech"}
    ],
}


def test_news_signal_valid():
    sig = NewsSignal.model_validate(_VALID_NEWS)
    assert sig.sector_sentiments["XLK"] == 0.8
    assert sig.sector_conviction["XLK"] == 0.75
    assert len(sig.key_themes) == 3


def test_news_signal_sentiment_out_of_range():
    bad = {**_VALID_NEWS, "sector_sentiments": {"XLK": 1.5}}
    with pytest.raises(ValidationError, match="out of"):
        NewsSignal.model_validate(bad)


def test_news_signal_conviction_out_of_range():
    bad = {**_VALID_NEWS, "sector_conviction": {"XLK": 1.1}}
    with pytest.raises(ValidationError, match="out of"):
        NewsSignal.model_validate(bad)


def test_news_signal_missing_field():
    with pytest.raises(ValidationError):
        NewsSignal.model_validate(
            {"sector_conviction": {"XLK": 0.5}, "key_themes": [], "evidence": []}
        )


# ---------------------------------------------------------------------------
# MacroRegimeSignal schema
# ---------------------------------------------------------------------------

_VALID_MACRO = {
    "reasoning": "VIX elevated and rising. Yield curve deeply inverted. FOMC hawkish language persists.",  # noqa: E501
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

_ALL_SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"]

_VALID_POLY = {
    "judgments": "No unusual correlations or context overrides.",
    "implied_probs": {"609655": 0.72, "612300": 0.35},
    "sector_tilts": {s: 0.0 for s in _ALL_SECTORS} | {"XLF": 0.5, "XLU": -0.3},
    "driving_events": [
        {
            "sector": "XLF",
            "market_question": "Fed rate cut by July 2026?",
            "reasoning": "Rate cut positive for financials",
        },
        {
            "sector": "XLU",
            "market_question": "US recession by end of 2026?",
            "reasoning": "Recession bearish for utilities",
        },
    ],
    "time_horizon": "short",
    "overall_confidence": 0.6,
}


def test_polymarket_signal_valid():
    sig = PolymarketSignal.model_validate(_VALID_POLY)
    assert sig.implied_probs["609655"] == 0.72
    assert sig.sector_tilts["XLF"] == 0.5
    assert sig.overall_confidence == 0.6
    assert any(e["sector"] == "XLF" for e in sig.driving_events)


def test_polymarket_signal_prob_out_of_range():
    bad = {**_VALID_POLY, "implied_probs": {"609655": 1.05}}
    with pytest.raises(ValidationError, match="out of"):
        PolymarketSignal.model_validate(bad)


def test_polymarket_signal_impact_out_of_range():
    bad = {**_VALID_POLY, "sector_tilts": {**_VALID_POLY["sector_tilts"], "XLF": -1.5}}
    with pytest.raises(ValidationError, match="out of"):
        PolymarketSignal.model_validate(bad)


def test_polymarket_signal_missing_time_horizon():
    bad = {k: v for k, v in _VALID_POLY.items() if k != "time_horizon"}
    with pytest.raises(ValidationError):
        PolymarketSignal.model_validate(bad)


def test_polymarket_signal_missing_overall_confidence():
    bad = {k: v for k, v in _VALID_POLY.items() if k != "overall_confidence"}
    with pytest.raises(ValidationError):
        PolymarketSignal.model_validate(bad)


# ---------------------------------------------------------------------------
# BaseAgent.validate_output
# ---------------------------------------------------------------------------


def test_validate_output_parses_valid_response(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    result = agent.validate_output(_tool_response(_VALID_NEWS))
    assert result["sector_conviction"]["XLK"] == 0.75
    assert result["sector_sentiments"]["XLK"] == 0.8


def test_validate_output_parses_attribute_style_tool_use_block(prompt_file):
    """validate_output handles both dict blocks and object-attribute blocks."""
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    block = MagicMock()
    block.type = "tool_use"
    block.input = _VALID_NEWS
    response = {"content": [block], "usage": {"input_tokens": 10, "output_tokens": 10}}
    result = agent.validate_output(response)
    assert result["sector_conviction"]["XLK"] == 0.75


def test_validate_output_raises_when_no_tool_use_block(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    text_only = {"content": [{"type": "text", "text": "some text, not a tool call"}], "usage": {}}
    with pytest.raises(ValueError, match="no tool_use content block"):
        agent.validate_output(text_only)


def test_validate_output_raises_on_empty_content(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    with pytest.raises(ValueError, match="no tool_use content block"):
        agent.validate_output({"content": [], "usage": {}})


def test_validate_output_raises_on_schema_violation(prompt_file):
    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file)
    bad_payload = {
        "sector_sentiments": {"XLK": 99.0},
        "sector_conviction": {"XLK": 0.5},
        "key_themes": [],
        "evidence": [],
    }
    with pytest.raises(ValueError):
        agent.validate_output(_tool_response(bad_payload))


# ---------------------------------------------------------------------------
# BaseAgent.run — end-to-end with mocked cached_call
# ---------------------------------------------------------------------------


def test_run_writes_signals_to_db(prompt_file):
    db = _make_engine()

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_VALID_NEWS)

    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file, cache=fake_cache)
    result = agent.run(datetime.date(2024, 1, 15), db)

    assert result["sector_conviction"]["XLK"] == 0.75

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

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_VALID_NEWS)

    agent = _FakeAgent("claude-haiku-4-5-20251001", prompt_file, cache=fake_cache)
    result = agent.run(datetime.date(2024, 1, 15), db)

    assert set(result.keys()) == {
        "sector_sentiments",
        "sector_conviction",
        "key_themes",
        "evidence",
    }


# ---------------------------------------------------------------------------
# NewsAgent
# ---------------------------------------------------------------------------

_FULL_NEWS_SIGNAL = {
    "sector_sentiments": {s: 0.0 for s in _ALL_SECTORS} | {"XLK": 0.6, "XLE": -0.4},
    "sector_conviction": {s: 0.1 for s in _ALL_SECTORS} | {"XLK": 0.55, "XLE": 0.5},
    "key_themes": [
        "AI spending accelerating",
        "oil supply surplus emerging",
        "rate cut expectations easing",
    ],
    "evidence": [
        {"sector": "XLK", "headline": "Nvidia Q1 beats on AI demand", "impact": "positive"},
        {"sector": "XLE", "headline": "OPEC+ agrees to output increase", "impact": "negative"},
    ],
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
    assert len(data["sectors"]["XLK"]) == 1
    assert data["sectors"]["XLU"] == []


def test_news_agent_prepare_input_ignores_articles_outside_window():
    from agents.news_agent import NewsAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
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

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_FULL_NEWS_SIGNAL, "report_sector_sentiment")

    agent = NewsAgent(cache=fake_cache)
    result = agent.run(date, db)

    assert result["sector_conviction"]["XLK"] == pytest.approx(0.55)

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

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_FULL_NEWS_SIGNAL, "report_sector_sentiment")

    agent = NewsAgent(cache=fake_cache)
    agent.run(date, db)
    agent.run(date, db)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="sentiment", date=date).all()

    assert len(rows) == 10


# ---------------------------------------------------------------------------
# MacroRegimeSignal schema
# ---------------------------------------------------------------------------

_VALID_MACRO_SIGNAL = {
    "reasoning": "VIX at 18.5, below 90d avg of 20.1 — mild risk-on. T10Y2Y at -0.3, improving from -0.5 avg. DGS10 fell 25bp in 30d signalling rate relief. CPI YoY 3.2%, declining trend. ICSA stable near 210k. Overall: cautious risk-on.",  # noqa: E501
    "regime": "neutral",
    "rate_outlook": "falling",
    "confidence": 0.68,
    "rationale": "Mixed but slightly improving macro backdrop. VIX easing and yield curve less inverted point toward neutral-to-risk-on, while still-elevated CPI prevents a full risk-on call. Rate outlook leans falling as the Fed approaches its terminal rate.",  # noqa: E501
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
        "VIXCLS": 18.5,
        "T10Y2Y": -0.35,
        "DGS10": 4.30,
        "DTWEXBGS": 103.5,
        "CPIAUCSL": 309.0,
        "UNRATE": 3.8,
        "ICSA": 210000,
    }

    rows = []
    for days_back in range(0, 95, 1):
        d = date - datetime.timedelta(days=days_back)
        for sid, val in series_values.items():
            rows.append(Macro(date=d, series_id=sid, value=val))

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


def test_macro_agent_prepare_input_no_series_30d():
    """series_30d was removed from the MacroAgent input to save ~1500 tokens per call."""
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_macro(db, date)

    agent = MacroAgent()
    data = agent.prepare_input(date, db)

    assert "series_30d" not in data


def test_macro_agent_prepare_input_empty_db_returns_empty_features():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    agent = MacroAgent()
    data = agent.prepare_input(datetime.date(2024, 6, 7), db)

    assert data["derived_features"] == {}
    assert "series_30d" not in data


def test_macro_agent_run_writes_2_signal_rows():
    from agents.macro_agent import MacroAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)
    _seed_macro(db, date)

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_VALID_MACRO_SIGNAL, "report_macro_regime")

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

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_VALID_MACRO_SIGNAL, "report_macro_regime")

    agent = MacroAgent(cache=fake_cache)
    agent.run(date, db)
    agent.run(date, db)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="macro", date=date).all()

    assert len(rows) == 2


# ---------------------------------------------------------------------------
# PolymarketAgent
# ---------------------------------------------------------------------------

_FULL_POLY_SIGNAL = {
    "judgments": "No unusual context. Volumes are sufficient on both markets.",
    "implied_probs": {"609655": 0.28, "1439536": 0.71},
    "sector_tilts": {s: 0.0 for s in _ALL_SECTORS}
    | {"XLK": 0.18, "XLF": -0.08, "XLY": 0.28, "XLRE": 0.18},
    "driving_events": [
        {
            "sector": "XLK",
            "market_question": "Fed rate cut by July 2026 meeting?",
            "reasoning": "Rate cuts boost tech valuations",
        },
        {
            "sector": "XLF",
            "market_question": "Fed rate cut by July 2026 meeting?",
            "reasoning": "Rate cuts compress net interest margins",
        },
        {
            "sector": "XLY",
            "market_question": "Fed rate cut by July 2026 meeting?",
            "reasoning": "Rate cuts boost consumer discretionary",
        },
        {
            "sector": "XLRE",
            "market_question": "Fed rate cut by July 2026 meeting?",
            "reasoning": "Rate cuts reduce REIT discount rates",
        },
    ],
    "time_horizon": "medium",
    "overall_confidence": 0.62,
}


def _seed_polymarket(db, market_ids: list[str], date: datetime.date) -> None:
    """Insert minimal polymarket_raw rows for given market_ids."""
    from db.models import PolymarketRaw

    ts_now = datetime.datetime.combine(date, datetime.time(12, 0))
    ts_30d = datetime.datetime.combine(date - datetime.timedelta(days=28), datetime.time(12, 0))
    end_dt = date + datetime.timedelta(days=60)

    with Session(db) as s:
        for mid in market_ids:
            s.add(
                PolymarketRaw(
                    market_id=mid,
                    timestamp=ts_30d,
                    question=f"Question for {mid}",
                    implied_prob=0.50,
                    volume=500000.0,
                    end_date=end_dt,
                )
            )
            s.add(
                PolymarketRaw(
                    market_id=mid,
                    timestamp=ts_now,
                    question=f"Question for {mid}",
                    implied_prob=0.60,
                    volume=600000.0,
                    end_date=end_dt,
                )
            )
        s.commit()


def test_polymarket_agent_uses_haiku_model():
    from agents.polymarket_agent import PolymarketAgent

    agent = PolymarketAgent()
    assert "haiku" in agent._model.lower()


def test_polymarket_agent_prepare_input_structure():
    from agents.polymarket_agent import PolymarketAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)

    agent = PolymarketAgent()
    market_ids = [m["market_id"] for m in agent._curated[:2]]
    _seed_polymarket(db, market_ids, date)

    data = agent.prepare_input(date, db)

    assert data["analysis_date"] == "2024-06-07"
    assert "markets" in data
    assert len(data["markets"]) == len(agent._curated)


def test_polymarket_agent_prepare_input_includes_prob_and_trend():
    from agents.polymarket_agent import PolymarketAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)

    agent = PolymarketAgent()
    market_ids = [m["market_id"] for m in agent._curated[:1]]
    _seed_polymarket(db, market_ids, date)

    data = agent.prepare_input(date, db)
    seeded = next(m for m in data["markets"] if m["market_id"] == market_ids[0])

    assert seeded["current_prob"] == pytest.approx(0.60)
    assert seeded["prob_30d_ago"] == pytest.approx(0.50)
    assert seeded["volume_usd"] == pytest.approx(600000.0)


def test_polymarket_agent_prepare_input_null_prob_when_no_db_row():
    from agents.polymarket_agent import PolymarketAgent

    db = _make_engine()
    agent = PolymarketAgent()
    data = agent.prepare_input(datetime.date(2024, 6, 7), db)

    for m in data["markets"]:
        assert m["current_prob"] is None


def test_polymarket_agent_run_writes_10_signal_rows():
    from agents.polymarket_agent import PolymarketAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_FULL_POLY_SIGNAL, "report_polymarket_tilts")

    agent = PolymarketAgent(cache=fake_cache)
    result = agent.run(date, db)

    assert result["time_horizon"] == "medium"
    assert result["overall_confidence"] == pytest.approx(0.62)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="events", date=date).all()

    assert len(rows) == 10
    targets = {r.target for r in rows}
    assert targets == set(_ALL_SECTORS)

    xlk = next(r for r in rows if r.target == "XLK")
    assert xlk.signal_value == pytest.approx(0.18)
    assert xlk.confidence == pytest.approx(0.62)


def test_polymarket_agent_run_is_idempotent():
    from agents.polymarket_agent import PolymarketAgent

    db = _make_engine()
    date = datetime.date(2024, 6, 7)

    def fake_cache(model, prompt, input_data, call_fn, *, agent_name, engine, tool=None):
        return _tool_response(_FULL_POLY_SIGNAL, "report_polymarket_tilts")

    agent = PolymarketAgent(cache=fake_cache)
    agent.run(date, db)
    agent.run(date, db)

    with Session(db) as s:
        rows = s.query(Signal).filter_by(agent_name="events", date=date).all()

    assert len(rows) == 10
