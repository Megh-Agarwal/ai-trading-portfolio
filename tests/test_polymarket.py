from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from db.models import Base, PolymarketRaw
from ingestion.polymarket import (
    _extract_yes_prob,
    _parse_end_date,
    fetch_active_markets,
    fetch_current_state,
    fetch_market_prices,
    load_curated_markets,
    write_polymarket,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def mock_requests():
    with patch("ingestion.polymarket.requests.get") as mock_get:
        yield mock_get


def _mock_response(json_data, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _market_dict(
    mid: str = "mkt-1",
    question: str = "Will X happen?",
    yes_price: str = "0.65",
    volume: str = "500000",
    end_date: str = "2026-12-31T00:00:00Z",
) -> dict:
    return {
        "conditionId": mid,
        "question": question,
        "outcomePrices": [yes_price, str(1 - float(yes_price))],
        "volume": volume,
        "endDate": end_date,
        "category": "economy",
    }


# ---------------------------------------------------------------------------
# fetch_active_markets
# ---------------------------------------------------------------------------


def test_fetch_active_markets_returns_list(mock_requests):
    mock_requests.return_value = _mock_response([_market_dict("m1"), _market_dict("m2")])
    result = fetch_active_markets(category="economy")
    assert len(result) == 2
    assert result[0]["conditionId"] == "m1"


def test_fetch_active_markets_passes_category_param(mock_requests):
    mock_requests.return_value = _mock_response([])
    fetch_active_markets(category="politics", limit=50)
    call_kwargs = mock_requests.call_args
    params = call_kwargs[1]["params"]
    assert params["tag"] == "politics"
    assert params["limit"] == 50


def test_fetch_active_markets_raises_on_http_error(mock_requests):
    import requests as req_lib

    resp = MagicMock()
    resp.raise_for_status.side_effect = req_lib.HTTPError("404")
    mock_requests.return_value = resp
    with pytest.raises(req_lib.HTTPError):
        fetch_active_markets()


# ---------------------------------------------------------------------------
# fetch_market_prices
# ---------------------------------------------------------------------------


def test_fetch_market_prices_returns_dataframe(mock_requests):
    mock_requests.return_value = _mock_response(
        {"history": [{"t": 1704067200, "p": "0.60"}, {"t": 1704153600, "p": "0.65"}]}
    )
    df = fetch_market_prices(
        "cond-id-123",
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 31),
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "price"]
    assert len(df) == 2
    assert df["price"].dtype == float


def test_fetch_market_prices_empty_history(mock_requests):
    mock_requests.return_value = _mock_response({"history": []})
    df = fetch_market_prices("cond", datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    assert df.empty
    assert list(df.columns) == ["timestamp", "price"]


def test_fetch_market_prices_returns_empty_on_network_error(mock_requests):
    import requests as req_lib

    mock_requests.side_effect = req_lib.ConnectionError("unreachable")
    df = fetch_market_prices("cond", datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    assert df.empty


def test_fetch_market_prices_sorted_ascending(mock_requests):
    mock_requests.return_value = _mock_response(
        {"history": [{"t": 1704153600, "p": "0.70"}, {"t": 1704067200, "p": "0.60"}]}
    )
    df = fetch_market_prices("c", datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    assert df["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# load_curated_markets
# ---------------------------------------------------------------------------


def test_load_curated_markets_from_tmp_yaml(tmp_path):
    content = {
        "markets": [
            {
                "market_id": "fed-cut",
                "question": "Will Fed cut?",
                "category": "economy",
                "confidence": "high",
                "sector_impacts": {"XLU": "positive_if_yes"},
            }
        ]
    }
    yaml_file = tmp_path / "markets.yaml"
    yaml_file.write_text(yaml.dump(content))
    markets = load_curated_markets(yaml_file)
    assert len(markets) == 1
    assert markets[0]["market_id"] == "fed-cut"
    assert markets[0]["sector_impacts"]["XLU"] == "positive_if_yes"


def test_load_curated_markets_empty_file(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("markets: []\n")
    assert load_curated_markets(yaml_file) == []


def test_load_curated_markets_real_yaml():
    """Acceptance test: real YAML has at least 10 entries with required fields."""
    real_path = Path(__file__).parent.parent / "config" / "polymarket_markets.yaml"
    markets = load_curated_markets(real_path)
    assert len(markets) >= 10
    for m in markets:
        assert "market_id" in m
        assert "question" in m
        assert "sector_impacts" in m
        assert len(m["sector_impacts"]) >= 1


# ---------------------------------------------------------------------------
# fetch_current_state
# ---------------------------------------------------------------------------


def test_fetch_current_state_normalizes_snapshot(mock_requests):
    curated = [{"market_id": "mkt-1", "question": "Will X?", "category": "economy"}]
    mock_requests.return_value = _mock_response(_market_dict("mkt-1", yes_price="0.72"))
    snapshots = fetch_current_state(curated)
    assert len(snapshots) == 1
    s = snapshots[0]
    assert s["market_id"] == "mkt-1"
    assert abs(s["implied_prob"] - 0.72) < 1e-6
    assert isinstance(s["timestamp"], datetime.datetime)
    assert isinstance(s["volume"], float)


def test_fetch_current_state_skips_failed_market(mock_requests):
    import requests as req_lib

    curated = [
        {"market_id": "gone", "question": "?"},
        {"market_id": "ok", "question": "?"},
    ]

    def side_effect(url, **kwargs):
        if "gone" in url:
            resp = MagicMock()
            resp.raise_for_status.side_effect = req_lib.HTTPError("404")
            return resp
        return _mock_response(_market_dict("ok"))

    mock_requests.side_effect = side_effect
    snapshots = fetch_current_state(curated)
    assert len(snapshots) == 1
    assert snapshots[0]["market_id"] == "ok"


def test_fetch_current_state_fallback_to_tokens_array(mock_requests):
    raw = {
        "question": "Will it rain?",
        "tokens": [
            {"outcome": "YES", "price": "0.55"},
            {"outcome": "NO", "price": "0.45"},
        ],
        "volume": "1000",
        "endDate": "2026-12-31T00:00:00Z",
        "category": "weather",
    }
    curated = [{"market_id": "rain", "question": "?"}]
    mock_requests.return_value = _mock_response(raw)
    snapshots = fetch_current_state(curated)
    assert abs(snapshots[0]["implied_prob"] - 0.55) < 1e-6


# ---------------------------------------------------------------------------
# write_polymarket
# ---------------------------------------------------------------------------


def test_write_polymarket_inserts_rows(engine):
    ts = datetime.datetime(2024, 1, 15, 12, 0)
    snapshots = [
        {
            "market_id": "m1",
            "timestamp": ts,
            "question": "Q1",
            "implied_prob": 0.65,
            "volume": 100000.0,
            "category": "economy",
            "end_date": datetime.date(2026, 12, 31),
        },
        {
            "market_id": "m2",
            "timestamp": ts,
            "question": "Q2",
            "implied_prob": 0.30,
            "volume": 50000.0,
            "category": "economy",
            "end_date": None,
        },
    ]
    n = write_polymarket(snapshots, engine)
    assert n == 2
    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(PolymarketRaw)).scalar()
    assert count == 2


def test_write_polymarket_upserts_on_conflict(engine):
    ts = datetime.datetime(2024, 1, 15, 12, 0)
    snap = {
        "market_id": "m1",
        "timestamp": ts,
        "question": "Q",
        "implied_prob": 0.50,
        "volume": 10000.0,
        "category": "economy",
        "end_date": None,
    }
    write_polymarket([snap], engine)
    snap["implied_prob"] = 0.75
    write_polymarket([snap], engine)
    with Session(engine) as session:
        row = session.execute(select(PolymarketRaw)).scalar_one()
    assert abs(row.implied_prob - 0.75) < 1e-6


def test_write_polymarket_empty_list_returns_zero(engine):
    assert write_polymarket([], engine) == 0


def test_write_polymarket_two_timestamps_same_market(engine):
    snap_base = {
        "market_id": "m1",
        "question": "Q",
        "implied_prob": 0.60,
        "volume": 5000.0,
        "category": "economy",
        "end_date": None,
    }
    s1 = {**snap_base, "timestamp": datetime.datetime(2024, 1, 10, 12)}
    s2 = {**snap_base, "timestamp": datetime.datetime(2024, 1, 11, 12), "implied_prob": 0.65}
    write_polymarket([s1, s2], engine)
    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(PolymarketRaw)).scalar()
    assert count == 2


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def test_extract_yes_prob_from_outcome_prices():
    raw = {"outcomePrices": ["0.72", "0.28"]}
    assert abs(_extract_yes_prob(raw) - 0.72) < 1e-6


def test_extract_yes_prob_from_tokens():
    raw = {"tokens": [{"outcome": "NO", "price": "0.35"}, {"outcome": "YES", "price": "0.65"}]}
    assert abs(_extract_yes_prob(raw) - 0.65) < 1e-6


def test_extract_yes_prob_returns_nan_when_missing():
    import math

    assert math.isnan(_extract_yes_prob({}))


def test_parse_end_date_various_formats():
    assert _parse_end_date("2026-12-31T00:00:00Z") == datetime.date(2026, 12, 31)
    assert _parse_end_date("2026-12-31T00:00:00.000Z") == datetime.date(2026, 12, 31)
    assert _parse_end_date("2026-12-31") == datetime.date(2026, 12, 31)
    assert _parse_end_date(None) is None
    assert _parse_end_date("not-a-date") is None
