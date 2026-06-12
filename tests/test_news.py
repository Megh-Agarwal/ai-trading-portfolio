from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from db.models import Base, NewsRaw
from ingestion.news import (
    aggregate_to_sector_week,
    fetch_company_news,
    validate_historical_depth,
    write_news,
)

START = datetime.date(2024, 1, 1)
END = datetime.date(2024, 1, 31)
TICKER = "AAPL"
SECTOR = "XLK"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finnhub_item(url: str, ticker: str = TICKER, ts: int = 1704153600) -> dict:
    return {
        "datetime": ts,
        "headline": f"News about {ticker}",
        "source": "Reuters",
        "summary": "Summary text.",
        "url": url,
        "related": ticker,
        "category": "company news",
        "id": hash(url),
        "image": "",
    }


@pytest.fixture()
def mock_finnhub(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test_key")
    with patch("ingestion.news.finnhub.Client") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        yield instance


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def populated_engine(engine):
    """Engine with a few pre-inserted news rows."""
    ts = datetime.datetime(2024, 1, 15, 10, 0)
    with Session(engine) as session:
        session.add_all([
            NewsRaw(ticker="AAPL", sector="XLK", timestamp=ts, source="Reuters",
                    title="Apple news 1", url="https://example.com/1"),
            NewsRaw(ticker="AAPL", sector="XLK", timestamp=ts, source="Reuters",
                    title="Apple news 2", url="https://example.com/2"),
            NewsRaw(ticker="MSFT", sector="XLK", timestamp=ts, source="Bloomberg",
                    title="Microsoft news", url="https://example.com/3"),
            # Outside the week window
            NewsRaw(ticker="AAPL", sector="XLK",
                    timestamp=datetime.datetime(2024, 2, 5), source="Reuters",
                    title="Feb news", url="https://example.com/4"),
        ])
        session.commit()
    return engine


# ---------------------------------------------------------------------------
# fetch_company_news
# ---------------------------------------------------------------------------


def test_fetch_company_news_returns_normalized_list(mock_finnhub):
    mock_finnhub.company_news.return_value = [
        _make_finnhub_item("https://example.com/a"),
        _make_finnhub_item("https://example.com/b"),
    ]
    result = fetch_company_news(TICKER, START, END)
    assert len(result) == 2
    assert all(k in result[0] for k in ("ticker", "timestamp", "source", "title", "summary", "url"))
    assert result[0]["ticker"] == TICKER
    assert isinstance(result[0]["timestamp"], datetime.datetime)


def test_fetch_company_news_empty_response(mock_finnhub):
    mock_finnhub.company_news.return_value = []
    assert fetch_company_news(TICKER, START, END) == []


def test_fetch_company_news_none_response(mock_finnhub):
    mock_finnhub.company_news.return_value = None
    assert fetch_company_news(TICKER, START, END) == []


def test_fetch_company_news_skips_items_without_datetime(mock_finnhub):
    mock_finnhub.company_news.return_value = [
        {"datetime": 0, "headline": "bad", "source": "x", "url": "u1"},
        _make_finnhub_item("https://example.com/good"),
    ]
    result = fetch_company_news(TICKER, START, END)
    # datetime=0 is falsy, should be skipped
    assert len(result) == 1


def test_fetch_company_news_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="FINNHUB_API_KEY"):
        fetch_company_news(TICKER, START, END)


# ---------------------------------------------------------------------------
# write_news — dedup by URL
# ---------------------------------------------------------------------------


def test_write_news_inserts_articles(mock_finnhub, engine):
    articles = [
        {"ticker": "AAPL", "timestamp": datetime.datetime(2024, 1, 10),
         "source": "Reuters", "title": "T1", "summary": "S1", "url": "https://u1.com"},
        {"ticker": "AAPL", "timestamp": datetime.datetime(2024, 1, 11),
         "source": "Reuters", "title": "T2", "summary": "S2", "url": "https://u2.com"},
    ]
    n = write_news(articles, sector=SECTOR, engine=engine)
    assert n == 2
    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(NewsRaw)).scalar()
    assert count == 2


def test_write_news_deduplicates_by_url(mock_finnhub, engine):
    article = {"ticker": "AAPL", "timestamp": datetime.datetime(2024, 1, 10),
               "source": "Reuters", "title": "T1", "summary": "S", "url": "https://u1.com"}
    write_news([article], sector=SECTOR, engine=engine)
    n = write_news([article], sector=SECTOR, engine=engine)  # second write
    assert n == 0
    with Session(engine) as session:
        count = session.execute(select(func.count()).select_from(NewsRaw)).scalar()
    assert count == 1


def test_write_news_tags_sector(engine):
    articles = [{"ticker": "AAPL", "timestamp": datetime.datetime(2024, 1, 10),
                 "source": "Reuters", "title": "T", "summary": "S", "url": "https://u.com"}]
    write_news(articles, sector="XLK", engine=engine)
    with Session(engine) as session:
        row = session.execute(select(NewsRaw)).scalar_one()
    assert row.sector == "XLK"


def test_write_news_empty_list_returns_zero(engine):
    assert write_news([], sector=SECTOR, engine=engine) == 0


# ---------------------------------------------------------------------------
# aggregate_to_sector_week
# ---------------------------------------------------------------------------


def test_aggregate_returns_expected_keys(populated_engine):
    holdings = {"XLK": ["AAPL", "MSFT"]}
    result = aggregate_to_sector_week("XLK", datetime.date(2024, 1, 8),
                                      populated_engine, holdings=holdings)
    assert set(result.keys()) >= {
        "etf", "week_start", "week_end", "article_count",
        "unique_url_count", "tickers_covered", "articles",
    }


def test_aggregate_filters_to_week(populated_engine):
    holdings = {"XLK": ["AAPL", "MSFT"]}
    # Articles are timestamped Jan 15; week_start Jan 15 covers Jan 15–21
    result = aggregate_to_sector_week("XLK", datetime.date(2024, 1, 15),
                                      populated_engine, holdings=holdings)
    assert result["article_count"] == 3


def test_aggregate_deduplicates_by_url(engine):
    ts = datetime.datetime(2024, 1, 10)
    with Session(engine) as session:
        session.add_all([
            NewsRaw(ticker="AAPL", sector="XLK", timestamp=ts,
                    title="Same story", url="https://dup.com"),
            NewsRaw(ticker="MSFT", sector="XLK", timestamp=ts,
                    title="Same story", url="https://dup.com"),  # same URL different ticker
        ])
        session.commit()
    holdings = {"XLK": ["AAPL", "MSFT"]}
    result = aggregate_to_sector_week("XLK", datetime.date(2024, 1, 8),
                                      engine, holdings=holdings)
    assert result["unique_url_count"] == 1
    assert result["article_count"] == 1


def test_aggregate_empty_when_no_news(engine):
    holdings = {"XLK": ["AAPL"]}
    result = aggregate_to_sector_week("XLK", datetime.date(2024, 1, 1),
                                      engine, holdings=holdings)
    assert result["article_count"] == 0
    assert result["articles"] == []


# ---------------------------------------------------------------------------
# validate_historical_depth
# ---------------------------------------------------------------------------


def test_validate_historical_depth(monkeypatch, mock_finnhub):
    # Simulate 13 months of news
    import time as time_mod
    base_ts = int(time_mod.time()) - (13 * 31 * 86400)
    mock_finnhub.company_news.return_value = [
        {"datetime": base_ts, "headline": "old", "source": "x", "url": "u", "summary": ""},
        {"datetime": int(time_mod.time()) - 86400, "headline": "recent", "source": "x",
         "url": "u2", "summary": ""},
    ]
    depth = validate_historical_depth(ticker="AAPL", api_key="test")
    assert depth >= 12
