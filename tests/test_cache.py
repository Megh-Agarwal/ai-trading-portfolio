from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import cache as cache_mod
from cache import cache_key, cached_call, get_cached, save_cached
from db.models import AgentCall, Base
from pricing import cost_usd, model_rates

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Redirect cache reads/writes to a temp directory."""
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _fake_response(tokens_in: int = 150, tokens_out: int = 40) -> dict:
    return {
        "id": "msg_test",
        "content": [{"type": "text", "text": "bullish"}],
        "model": "claude-haiku-4-5-20251001",
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic():
    k1 = cache_key("claude-haiku-4-5-20251001", "system prompt", {"sector": "XLK"})
    k2 = cache_key("claude-haiku-4-5-20251001", "system prompt", {"sector": "XLK"})
    assert k1 == k2


def test_cache_key_is_64_hex_chars():
    k = cache_key("model", "prompt", {})
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_cache_key_changes_with_model():
    k1 = cache_key("claude-haiku-4-5-20251001", "p", {"x": 1})
    k2 = cache_key("claude-sonnet-4-6", "p", {"x": 1})
    assert k1 != k2


def test_cache_key_changes_with_prompt():
    k1 = cache_key("m", "prompt A", {"x": 1})
    k2 = cache_key("m", "prompt B", {"x": 1})
    assert k1 != k2


def test_cache_key_changes_with_input_data():
    k1 = cache_key("m", "p", {"sector": "XLK"})
    k2 = cache_key("m", "p", {"sector": "XLF"})
    assert k1 != k2


def test_cache_key_input_data_order_invariant():
    k1 = cache_key("m", "p", {"a": 1, "b": 2})
    k2 = cache_key("m", "p", {"b": 2, "a": 1})
    assert k1 == k2


# ---------------------------------------------------------------------------
# get_cached / save_cached
# ---------------------------------------------------------------------------


def test_get_cached_returns_none_on_miss(cache_dir):
    assert get_cached("nonexistent" * 4) is None


def test_save_and_get_roundtrip(cache_dir):
    response = _fake_response()
    key = cache_key("m", "p", {})
    save_cached(key, response)
    assert get_cached(key) == response


def test_get_cached_returns_none_on_corrupt_json(cache_dir):
    key = cache_key("m", "p", {})
    (cache_dir / f"{key}.json").write_text("not-valid-json")
    assert get_cached(key) is None


def test_save_cached_creates_cache_dir(tmp_path, monkeypatch):
    nested = tmp_path / "new" / "cache"
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", nested)
    assert not nested.exists()
    save_cached(cache_key("m", "p", {}), _fake_response())
    assert nested.exists()


# ---------------------------------------------------------------------------
# cached_call — miss path
# ---------------------------------------------------------------------------


def test_cached_call_miss_invokes_call_fn(cache_dir):
    call_fn = MagicMock(return_value=_fake_response())
    cached_call("claude-haiku-4-5-20251001", "prompt", {"x": 1}, call_fn)
    call_fn.assert_called_once()


def test_cached_call_miss_returns_call_fn_result(cache_dir):
    expected = _fake_response(tokens_in=200, tokens_out=50)
    result = cached_call("claude-haiku-4-5-20251001", "p", {}, lambda: expected)
    assert result == expected


def test_cached_call_miss_writes_to_disk(cache_dir):
    cached_call("model", "prompt", {}, lambda: _fake_response())
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# cached_call — hit path
# ---------------------------------------------------------------------------


def test_cached_call_hit_does_not_invoke_call_fn(cache_dir):
    call_fn = MagicMock(return_value=_fake_response())
    cached_call("claude-haiku-4-5-20251001", "prompt", {"x": 1}, call_fn)
    assert call_fn.call_count == 1  # first call: miss

    call_fn2 = MagicMock(return_value=_fake_response())
    result = cached_call("claude-haiku-4-5-20251001", "prompt", {"x": 1}, call_fn2)
    call_fn2.assert_not_called()  # second call: hit
    assert result["id"] == "msg_test"


def test_cached_call_hit_returns_same_data(cache_dir):
    resp = _fake_response(tokens_in=111, tokens_out=22)
    call_fn = MagicMock(return_value=resp)
    cached_call("m", "p", {"k": "v"}, call_fn)
    result = cached_call("m", "p", {"k": "v"}, call_fn)
    assert result == resp


# ---------------------------------------------------------------------------
# cached_call — DB logging
# ---------------------------------------------------------------------------


def test_cached_call_miss_logs_to_agent_calls(cache_dir, engine):
    cached_call(
        "claude-haiku-4-5-20251001",
        "system prompt",
        {"tickers": ["AAPL"]},
        lambda: _fake_response(100, 30),
        agent_name="sentiment",
        engine=engine,
    )
    with Session(engine) as session:
        rows = session.execute(select(AgentCall)).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.agent_name == "sentiment"
    assert row.model_string == "claude-haiku-4-5-20251001"
    assert row.tokens_in == 100
    assert row.tokens_out == 30
    assert row.cached is False
    assert row.cost_usd is not None and row.cost_usd > 0


def test_cached_call_hit_logs_zero_cost_row(cache_dir, engine):
    model = "claude-haiku-4-5-20251001"
    call_fn = MagicMock(return_value=_fake_response(100, 30))
    # First call — miss
    cached_call(model, "p", {}, call_fn, agent_name="a", engine=engine)
    # Second call — hit
    cached_call(model, "p", {}, call_fn, agent_name="a", engine=engine)

    with Session(engine) as session:
        rows = session.execute(select(AgentCall).order_by(AgentCall.call_id)).scalars().all()

    assert len(rows) == 2
    miss_row, hit_row = rows
    assert miss_row.cached is False
    assert hit_row.cached is True
    assert hit_row.tokens_in == 0
    assert hit_row.tokens_out == 0
    assert hit_row.cost_usd == 0.0


def test_cached_call_no_engine_skips_db(cache_dir):
    result = cached_call("model", "p", {}, lambda: _fake_response(), engine=None)
    assert result["id"] == "msg_test"


def test_cached_call_logs_prompt_hash_and_input_hash(cache_dir, engine):
    import hashlib

    prompt = "this is a system prompt"
    input_data = {"sector": "XLK"}
    cached_call(
        "claude-haiku-4-5-20251001",
        prompt,
        input_data,
        lambda: _fake_response(),
        engine=engine,
    )
    with Session(engine) as session:
        row = session.execute(select(AgentCall)).scalar_one()

    expected_prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    expected_input_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()
    assert row.prompt_hash == expected_prompt_hash
    assert row.input_hash == expected_input_hash


def test_cached_call_records_latency_ms(cache_dir, engine):
    cached_call("claude-sonnet-4-6", "p", {}, lambda: _fake_response(), engine=engine)
    with Session(engine) as session:
        row = session.execute(select(AgentCall)).scalar_one()
    assert row.latency_ms is not None
    assert row.latency_ms >= 0


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------


def test_cost_usd_haiku_known_tokens():
    # 1M in + 0 out = $0.80; so 1000 in + 500 out = 0.000800 + 0.002000 = $0.002800
    c = cost_usd("claude-haiku-4-5-20251001", 1000, 500)
    assert abs(c - (1000 * 0.80 + 500 * 4.00) / 1_000_000) < 1e-9


def test_cost_usd_sonnet_known_tokens():
    c = cost_usd("claude-sonnet-4-6", 500, 200)
    assert abs(c - (500 * 3.00 + 200 * 15.00) / 1_000_000) < 1e-9


def test_cost_usd_zero_tokens():
    assert cost_usd("claude-haiku-4-5-20251001", 0, 0) == 0.0


def test_cost_usd_unknown_model_returns_zero():
    assert cost_usd("gpt-4-unknown", 1000, 500) == 0.0


def test_cost_usd_prefix_match():
    # A hypothetical future variant should still match
    c = cost_usd("claude-haiku-4-5-some-future-suffix", 1000, 0)
    assert abs(c - 1000 * 0.80 / 1_000_000) < 1e-9


def test_model_rates_known():
    rates = model_rates("claude-sonnet-4-6")
    assert rates is not None
    assert rates == (3.00, 15.00)


def test_model_rates_unknown():
    assert model_rates("gpt-4") is None
