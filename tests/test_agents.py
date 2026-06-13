"""Smoke tests for BaseAgent and the three agent output schemas."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
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
