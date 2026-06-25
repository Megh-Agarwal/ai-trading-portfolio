from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from db.models import Base, NewsRaw
from ingestion.alpha_vantage_news import (
    fetch_av_news,
    validate_av_historical_depth,
    write_av_news,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T_FROM = datetime.datetime(2025, 1, 1)
_T_TO = datetime.datetime(2026, 1, 1)

_ARTICLE_AAPL = {
    "title": "Apple hits record",
    "url": "https://example.com/apple-record",
    "time_published": "20250615T120000",
    "summary": "Apple shares hit a record high.",
    "source": "Reuters",
    "ticker_sentiment": [
        {"ticker": "AAPL", "relevance_score": "0.9"},
        {"ticker": "MSFT", "relevance_score": "0.2"},
    ],
}

_ARTICLE_MSFT_ONLY = {
    "title": "Microsoft cloud beats",
    "url": "https://example.com/msft-cloud",
    "time_published": "20250620T090000",
    "summary": "Azure revenue up.",
    "source": "Bloomberg",
    "ticker_sentiment": [
        {"ticker": "MSFT", "relevance_score": "0.95"},
    ],
}

_ARTICLE_UNRELATED = {
    "title": "Oil prices spike",
    "url": "https://example.com/oil-spike",
    "time_published": "20250701T080000",
    "summary": "Crude up 5%.",
    "source": "WSJ",
    "ticker_sentiment": [
        {"ticker": "XOM", "relevance_score": "0.8"},
    ],
}


def _make_av_response(feed: list[dict]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"feed": feed}
    return mock_resp


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# TestFetchAvNews
# ---------------------------------------------------------------------------


class TestFetchAvNews:
    def test_returns_articles_for_matched_tickers(self):
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([_ARTICLE_AAPL])
            articles = fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

        assert len(articles) == 1
        assert articles[0]["ticker"] == "AAPL"
        assert articles[0]["title"] == "Apple hits record"
        assert articles[0]["url"] == "https://example.com/apple-record"
        assert isinstance(articles[0]["timestamp"], datetime.datetime)

    def test_fans_out_article_to_multiple_queried_tickers(self):
        # Both AAPL and MSFT appear in ticker_sentiment; querying both → two rows
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([_ARTICLE_AAPL])
            articles = fetch_av_news(["AAPL", "MSFT"], _T_FROM, _T_TO, api_key="test")

        tickers_returned = [a["ticker"] for a in articles]
        assert "AAPL" in tickers_returned
        assert "MSFT" in tickers_returned
        assert len(articles) == 2
        # Both rows share the same url
        assert all(a["url"] == "https://example.com/apple-record" for a in articles)

    def test_excludes_tickers_not_in_query_set(self):
        # Query only AAPL; MSFT also appears in ticker_sentiment but should be excluded
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([_ARTICLE_AAPL])
            articles = fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

        assert all(a["ticker"] == "AAPL" for a in articles)

    def test_skips_article_whose_ticker_sentiment_contains_no_queried_tickers(self):
        # Article has ticker_sentiment=[XOM] but we queried AAPL — AV says it's about XOM, skip it.
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([_ARTICLE_UNRELATED])
            articles = fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

        assert articles == []

    def test_attributes_article_to_queried_ticker_when_ticker_sentiment_is_empty(self):
        # AV returned this article for our AAPL query but ticker_sentiment is absent.
        # Fall back: attribute to the queried ticker.
        article_no_sentiment = {**_ARTICLE_AAPL, "ticker_sentiment": []}
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([article_no_sentiment])
            articles = fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

        assert len(articles) == 1
        assert articles[0]["ticker"] == "AAPL"

    def test_handles_empty_feed(self):
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([])
            articles = fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

        assert articles == []

    def test_raises_on_information_key(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"Information": "API call frequency limit reached."}
        with (
            patch("ingestion.alpha_vantage_news.requests.get", return_value=mock_resp),
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            with pytest.raises(RuntimeError, match="Alpha Vantage API error"):
                fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ALPHA_VANTAGE_API_KEY"):
            fetch_av_news(["AAPL"], _T_FROM, _T_TO)

    def test_raises_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        with (
            patch("ingestion.alpha_vantage_news.requests.get", return_value=mock_resp),
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            with pytest.raises(Exception, match="500"):
                fetch_av_news(["AAPL"], _T_FROM, _T_TO, api_key="test")

    def test_skips_article_with_unparseable_timestamp(self):
        bad_article = {**_ARTICLE_AAPL, "time_published": "not-a-date"}
        with (
            patch("ingestion.alpha_vantage_news.requests.get") as mock_get,
            patch("ingestion.alpha_vantage_news._rate_limit"),
        ):
            mock_get.return_value = _make_av_response([bad_article, _ARTICLE_MSFT_ONLY])
            articles = fetch_av_news(["AAPL", "MSFT"], _T_FROM, _T_TO, api_key="test")

        # Only MSFT article (valid timestamp) should be returned
        assert len(articles) == 1
        assert articles[0]["ticker"] == "MSFT"


# ---------------------------------------------------------------------------
# TestWriteAvNews
# ---------------------------------------------------------------------------


class TestWriteAvNews:
    def _make_article(self, ticker: str, url: str) -> dict:
        return {
            "ticker": ticker,
            "timestamp": datetime.datetime(2025, 6, 15, 12, 0),
            "source": "Reuters",
            "title": f"News about {ticker}",
            "summary": "Some summary.",
            "url": url,
        }

    def test_inserts_new_articles(self, engine):
        articles = [
            self._make_article("AAPL", "https://example.com/a1"),
            self._make_article("MSFT", "https://example.com/a2"),
        ]
        n = write_av_news(articles, sector="XLK", engine=engine)
        assert n == 2

    def test_skips_duplicate_url_ticker_pair(self, engine):
        article = self._make_article("AAPL", "https://example.com/a1")
        write_av_news([article], sector="XLK", engine=engine)
        # Second write with same url+ticker → 0 new rows
        n = write_av_news([article], sector="XLK", engine=engine)
        assert n == 0

    def test_allows_same_url_for_different_tickers(self, engine):
        # Key difference vs Finnhub: same URL with different ticker IS a new row
        a1 = self._make_article("AAPL", "https://example.com/shared")
        a2 = self._make_article("MSFT", "https://example.com/shared")
        write_av_news([a1], sector="XLK", engine=engine)
        n = write_av_news([a2], sector="XLK", engine=engine)
        assert n == 1

        with Session(engine) as session:
            rows = (
                session.execute(select(NewsRaw).where(NewsRaw.url == "https://example.com/shared"))
                .scalars()
                .all()
            )
        assert len(rows) == 2
        assert {r.ticker for r in rows} == {"AAPL", "MSFT"}

    def test_returns_zero_on_empty_input(self, engine):
        assert write_av_news([], sector="XLK", engine=engine) == 0

    def test_sets_sector_on_all_rows(self, engine):
        articles = [
            self._make_article("AAPL", "https://example.com/s1"),
            self._make_article("MSFT", "https://example.com/s2"),
        ]
        write_av_news(articles, sector="XLK", engine=engine)
        with Session(engine) as session:
            rows = session.execute(select(NewsRaw)).scalars().all()
        assert all(r.sector == "XLK" for r in rows)


# ---------------------------------------------------------------------------
# TestValidateAvHistoricalDepth
# ---------------------------------------------------------------------------


class TestValidateAvHistoricalDepth:
    def test_returns_months_of_history(self):
        # Simulate AV returning an article from ~6 months ago
        six_months_ago = datetime.datetime.now() - datetime.timedelta(days=185)
        fake_article = {
            "ticker": "AAPL",
            "timestamp": six_months_ago,
            "source": "Reuters",
            "title": "Old news",
            "summary": None,
            "url": "https://example.com/old",
        }
        with patch("ingestion.alpha_vantage_news.fetch_av_news", return_value=[fake_article]):
            depth = validate_av_historical_depth(ticker="AAPL", api_key="test")

        assert depth == 6

    def test_returns_zero_when_no_articles(self):
        with patch("ingestion.alpha_vantage_news.fetch_av_news", return_value=[]):
            depth = validate_av_historical_depth(ticker="AAPL", api_key="test")

        assert depth == 0
