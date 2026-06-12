from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from db.models import Base, Macro
from ingestion.macro import fetch_macro, write_macro

START = datetime.date(2024, 1, 1)
END = datetime.date(2024, 3, 31)
SERIES = ["T10Y2Y", "DGS10", "VIXCLS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_series(start: datetime.date, end: datetime.date, value: float = 1.5) -> pd.Series:
    idx = pd.date_range(start, end, freq="D")
    return pd.Series(value, index=idx)


def _monthly_series(start: datetime.date, end: datetime.date, value: float = 3.2) -> pd.Series:
    idx = pd.date_range(start, end, freq="MS")
    return pd.Series(value, index=idx)


def _weekly_series(start: datetime.date, end: datetime.date, value: float = 210_000) -> pd.Series:
    idx = pd.date_range(start, end, freq="W-THU")
    return pd.Series(value, index=idx)


@pytest.fixture()
def mock_fred():
    with patch("ingestion.macro.Fred") as MockFred:
        instance = MagicMock()
        MockFred.return_value = instance
        yield instance


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# fetch_macro — happy path
# ---------------------------------------------------------------------------


def test_fetch_macro_returns_expected_columns(mock_fred, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test")
    mock_fred.get_series.return_value = _daily_series(START, END)

    result = fetch_macro(SERIES, START, END)

    assert set(result.columns) == {"date", "series_id", "value"}
    assert set(result["series_id"].unique()) == set(SERIES)
    assert all(isinstance(d, datetime.date) for d in result["date"])


def test_fetch_macro_daily_series_has_no_nan(mock_fred, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test")
    mock_fred.get_series.return_value = _daily_series(START, END)

    result = fetch_macro(["T10Y2Y"], START, END)

    assert result["value"].isna().sum() == 0


def test_fetch_macro_monthly_series_forward_filled_to_daily(mock_fred, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test")
    # Monthly series starting before START (simulates buffer working correctly)
    fetch_start = START - datetime.timedelta(days=90)
    mock_fred.get_series.return_value = _monthly_series(fetch_start, END, value=3.5)

    result = fetch_macro(["CPIAUCSL"], START, END)

    expected_days = (END - START).days + 1
    assert len(result) == expected_days
    assert result["value"].isna().sum() == 0
    assert (result["value"] == 3.5).all()


def test_fetch_macro_weekly_series_forward_filled_to_daily(mock_fred, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test")
    fetch_start = START - datetime.timedelta(days=90)
    mock_fred.get_series.return_value = _weekly_series(fetch_start, END, value=210_000)

    result = fetch_macro(["ICSA"], START, END)

    expected_days = (END - START).days + 1
    assert len(result) == expected_days
    assert result["value"].isna().sum() == 0


def test_fetch_macro_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
        fetch_macro(["T10Y2Y"], START, END)


def test_fetch_macro_empty_series_skipped(mock_fred, monkeypatch, caplog):
    monkeypatch.setenv("FRED_API_KEY", "test")
    mock_fred.get_series.return_value = pd.Series(dtype=float)

    with caplog.at_level("WARNING", logger="ingestion.macro"):
        result = fetch_macro(["T10Y2Y"], START, END)

    assert result.empty
    assert any("skipping" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# write_macro — upsert semantics
# ---------------------------------------------------------------------------


def test_write_macro_inserts_rows(mock_fred, monkeypatch, engine):
    monkeypatch.setenv("FRED_API_KEY", "test")
    mock_fred.get_series.return_value = _daily_series(START, END)

    df = fetch_macro(["T10Y2Y"], START, END)
    n = write_macro(df, engine)

    assert n == len(df)
    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(Macro)).scalar()
    assert count == len(df)


def test_write_macro_is_idempotent(mock_fred, monkeypatch, engine):
    monkeypatch.setenv("FRED_API_KEY", "test")
    mock_fred.get_series.return_value = _daily_series(START, END)

    df = fetch_macro(["T10Y2Y"], START, END)
    write_macro(df, engine)
    write_macro(df, engine)

    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(Macro)).scalar()
    assert count == len(df)


def test_write_macro_updates_existing_value(mock_fred, monkeypatch, engine):
    monkeypatch.setenv("FRED_API_KEY", "test")
    mock_fred.get_series.return_value = _daily_series(START, END, value=1.0)
    df = fetch_macro(["DGS10"], START, END)
    write_macro(df, engine)

    # Revised value — simulates a FRED data revision
    mock_fred.get_series.return_value = _daily_series(START, END, value=2.0)
    df2 = fetch_macro(["DGS10"], START, END)
    write_macro(df2, engine)

    with Session(engine) as session:
        row = session.get(Macro, (START, "DGS10"))
    assert row is not None
    assert row.value == pytest.approx(2.0)
