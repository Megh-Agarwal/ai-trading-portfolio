from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from ingestion.holdings import fetch_top_holdings, load_holdings, validate_holdings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_holdings_df(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [f"Company {s}" for s in symbols],
            "Holding Percent": [0.1 - i * 0.005 for i in range(len(symbols))],
        },
        index=pd.Index(symbols, name="Symbol"),
    )


@pytest.fixture()
def mock_yf_ticker():
    symbols = [f"TICK{i}" for i in range(10)]
    with patch("ingestion.holdings.yf.Ticker") as MockTicker:
        instance = MagicMock()
        instance.funds_data.top_holdings = _make_holdings_df(symbols)
        MockTicker.return_value = instance
        yield MockTicker


# ---------------------------------------------------------------------------
# fetch_top_holdings
# ---------------------------------------------------------------------------


def test_fetch_top_holdings_returns_list_of_strings(mock_yf_ticker):
    result = fetch_top_holdings("XLK")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


def test_fetch_top_holdings_default_n_is_10(mock_yf_ticker):
    result = fetch_top_holdings("XLK")
    assert len(result) == 10


def test_fetch_top_holdings_respects_n(mock_yf_ticker):
    result = fetch_top_holdings("XLK", n=5)
    assert len(result) == 5


def test_fetch_top_holdings_passes_ticker_to_yfinance(mock_yf_ticker):
    fetch_top_holdings("XLF")
    mock_yf_ticker.assert_called_once_with("XLF")


# ---------------------------------------------------------------------------
# validate_holdings
# ---------------------------------------------------------------------------


def test_validate_holdings_no_overlap():
    holdings = {"XLK": ["AAPL", "MSFT", "NVDA"], "XLF": ["JPM", "BAC", "V"]}
    assert validate_holdings(holdings) == {}


def test_validate_holdings_detects_overlap():
    holdings = {"XLK": ["AAPL", "MSFT"], "XLY": ["AAPL", "AMZN"]}
    overlaps = validate_holdings(holdings)
    assert "AAPL" in overlaps
    assert set(overlaps["AAPL"]) == {"XLK", "XLY"}


def test_validate_holdings_only_flags_multi_sector():
    holdings = {"XLK": ["AAPL", "MSFT"], "XLF": ["JPM"], "XLY": ["AMZN"]}
    overlaps = validate_holdings(holdings)
    assert overlaps == {}


# ---------------------------------------------------------------------------
# load_holdings
# ---------------------------------------------------------------------------


def test_load_holdings_returns_expected_structure(tmp_path):
    data = {"XLK": ["AAPL", "MSFT"], "XLF": ["JPM", "BAC"]}
    (tmp_path / "sector_holdings.yaml").write_text(yaml.dump(data))
    result = load_holdings(tmp_path / "sector_holdings.yaml")
    assert result == data


def test_load_holdings_strips_non_list_keys(tmp_path):
    data = {"XLK": ["AAPL", "MSFT"], "_meta": "some string"}
    (tmp_path / "sector_holdings.yaml").write_text(yaml.dump(data))
    result = load_holdings(tmp_path / "sector_holdings.yaml")
    assert "_meta" not in result
    assert "XLK" in result


def test_load_holdings_raises_if_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="update_holdings"):
        load_holdings(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# Acceptance: config/sector_holdings.yaml has 100 tickers, 10 per sector
# ---------------------------------------------------------------------------


def test_sector_holdings_yaml_has_100_tickers():
    yaml_path = Path(__file__).parent.parent / "config" / "sector_holdings.yaml"
    if not yaml_path.exists():
        pytest.skip("Run scripts/update_holdings.py first to generate the cache")
    holdings = load_holdings(yaml_path)
    assert len(holdings) == 10, f"Expected 10 ETFs, got {len(holdings)}"
    for etf, tickers in holdings.items():
        assert len(tickers) == 10, f"{etf} has {len(tickers)} holdings, expected 10"


def test_sector_holdings_yaml_no_overlaps():
    yaml_path = Path(__file__).parent.parent / "config" / "sector_holdings.yaml"
    if not yaml_path.exists():
        pytest.skip("Run scripts/update_holdings.py first to generate the cache")
    holdings = load_holdings(yaml_path)
    overlaps = validate_holdings(holdings)
    assert overlaps == {}, f"Unexpected cross-sector overlaps: {overlaps}"
