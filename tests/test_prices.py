from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from db.models import Base, Price
from ingestion.prices import _check_quality, _reshape, fetch_prices, write_prices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yf_download(tickers: list[str], dates: list[str]) -> pd.DataFrame:
    """Build a mock yfinance.download() return value (MultiIndex columns)."""
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    col_idx = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    idx = pd.DatetimeIndex(dates, name="Date")

    data = {}
    for field in fields:
        for ticker in tickers:
            base = 100.0 + tickers.index(ticker) * 10
            if field == "Volume":
                data[(field, ticker)] = [1_000_000 + i * 100_000 for i in range(len(dates))]
            elif field == "Adj Close":
                data[(field, ticker)] = [base * 0.98 + i * 0.5 for i in range(len(dates))]
            else:
                data[(field, ticker)] = [base + i * 0.5 for i in range(len(dates))]

    return pd.DataFrame(data, index=idx, columns=col_idx)


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


TICKERS = ["XLK", "XLF"]
DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]


# ---------------------------------------------------------------------------
# fetch_prices
# ---------------------------------------------------------------------------


def test_fetch_prices_returns_expected_columns():
    mock_df = _make_yf_download(TICKERS, DATES)
    with patch("ingestion.prices.yf.download", return_value=mock_df):
        result = fetch_prices(TICKERS, datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))

    expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    assert set(result.columns) >= expected_cols
    assert len(result) == len(TICKERS) * len(DATES)
    assert set(result["ticker"].unique()) == set(TICKERS)
    assert all(isinstance(d, datetime.date) for d in result["date"])


def test_fetch_prices_empty_download_returns_empty_df():
    with patch("ingestion.prices.yf.download", return_value=pd.DataFrame()):
        result = fetch_prices(TICKERS, datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))
    assert result.empty


# ---------------------------------------------------------------------------
# _check_quality (via caplog)
# ---------------------------------------------------------------------------


def test_check_quality_warns_on_nan_close(caplog):
    df = _reshape(_make_yf_download(TICKERS, DATES), TICKERS)
    df.loc[df["ticker"] == "XLK", "close"] = float("nan")

    with caplog.at_level("WARNING", logger="ingestion.prices"):
        _check_quality(df)

    assert any("NaN close" in msg for msg in caplog.messages)


def test_check_quality_warns_on_zero_close(caplog):
    df = _reshape(_make_yf_download(TICKERS, DATES), TICKERS)
    df.loc[(df["ticker"] == "XLF") & (df["date"] == df["date"].iloc[0]), "close"] = 0.0

    with caplog.at_level("WARNING", logger="ingestion.prices"):
        _check_quality(df)

    assert any("zero/negative close" in msg for msg in caplog.messages)


def test_check_quality_warns_on_zero_adj_close(caplog):
    df = _reshape(_make_yf_download(TICKERS, DATES), TICKERS)
    df.loc[df["ticker"] == "XLK", "adj_close"] = -1.0

    with caplog.at_level("WARNING", logger="ingestion.prices"):
        _check_quality(df)

    assert any("zero/negative adj_close" in msg for msg in caplog.messages)


def test_check_quality_no_warnings_on_clean_data(caplog):
    df = _reshape(_make_yf_download(TICKERS, DATES), TICKERS)
    with caplog.at_level("WARNING", logger="ingestion.prices"):
        _check_quality(df)
    assert not any("QUALITY" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# write_prices — upsert semantics
# ---------------------------------------------------------------------------


def test_write_prices_inserts_rows(engine):
    mock_df = _make_yf_download(TICKERS, DATES)
    with patch("ingestion.prices.yf.download", return_value=mock_df):
        df = fetch_prices(TICKERS, datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))

    n = write_prices(df, engine)
    assert n == len(TICKERS) * len(DATES)

    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(Price)).scalar()
    assert count == len(TICKERS) * len(DATES)


def test_write_prices_is_idempotent(engine):
    mock_df = _make_yf_download(TICKERS, DATES)
    with patch("ingestion.prices.yf.download", return_value=mock_df):
        df = fetch_prices(TICKERS, datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))

    write_prices(df, engine)
    write_prices(df, engine)  # second write must not duplicate

    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(Price)).scalar()
    assert count == len(TICKERS) * len(DATES)


def test_write_prices_updates_existing_value(engine):
    mock_df = _make_yf_download(TICKERS, DATES)
    with patch("ingestion.prices.yf.download", return_value=mock_df):
        df = fetch_prices(TICKERS, datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))

    write_prices(df, engine)

    # Modify close for XLK on first date and write again
    df.loc[(df["ticker"] == "XLK") & (df["date"] == datetime.date(2024, 1, 2)), "close"] = 999.0
    write_prices(df, engine)

    with Session(engine) as session:
        row = session.get(Price, (datetime.date(2024, 1, 2), "XLK"))
    assert row is not None
    assert row.close == pytest.approx(999.0)
