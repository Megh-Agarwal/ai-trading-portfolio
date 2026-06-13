from __future__ import annotations

import datetime
import logging
import os
import time

import requests
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from db.models import NewsRaw

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
# Free tier: ~5 calls/min. 12s gives comfortable headroom.
_CALL_INTERVAL = 12.0
_last_call_ts: float = 0.0


def fetch_av_news(
    tickers: list[str],
    time_from: datetime.datetime,
    time_to: datetime.datetime,
    limit: int = 200,
    api_key: str | None = None,
) -> list[dict]:
    """Fetch news from Alpha Vantage NEWS_SENTIMENT for a batch of tickers.

    One API call covers all tickers in the list. Articles are fanned out:
    one dict per (article, ticker) pair where the ticker appears in the
    article's ticker_sentiment list. Only tickers in the queried set are
    included — unrelated tickers mentioned in the article are ignored.

    Args:
        tickers: Stock symbols to query (batched into one API call).
        time_from: Start of date range (inclusive).
        time_to: End of date range (inclusive).
        limit: Max articles per response (200 on free tier, 1000 on premium).
        api_key: Alpha Vantage key. Defaults to ALPHA_VANTAGE_API_KEY env var.

    Returns:
        List of normalized article dicts with keys:
        ticker, timestamp, source, title, summary, url.
    """
    key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise EnvironmentError("ALPHA_VANTAGE_API_KEY not set. Add it to .env or export it.")

    _rate_limit()

    # AV requires literal commas in the tickers param. requests.get(params=) encodes
    # them as %2C, which AV silently treats as a single invalid ticker → empty feed.
    # Fix: pass everything except tickers via params (safely encoded), append tickers raw.
    params = {
        "function": "NEWS_SENTIMENT",
        "time_from": time_from.strftime("%Y%m%dT%H%M"),
        "time_to": time_to.strftime("%Y%m%dT%H%M"),
        "limit": limit,
        "sort": "EARLIEST",
        "apikey": key,
    }

    req = requests.Request("GET", _BASE_URL, params=params)
    url = req.prepare().url + "&tickers=" + ",".join(tickers)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # AV signals rate limits and auth errors via these keys instead of HTTP status codes
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage API error: {data['Information']}")
    if "Note" in data:
        logger.warning("Alpha Vantage rate limit warning: %s", data["Note"])

    ticker_set = set(tickers)
    articles: list[dict] = []

    for item in data.get("feed", []):
        raw_ts = item.get("time_published", "")
        if not raw_ts:
            continue
        ts = _parse_av_timestamp(raw_ts)
        if ts is None:
            logger.debug("Unparseable timestamp %r — skipping article", raw_ts)
            continue

        url = item.get("url")
        title = item.get("title") or ""
        summary = item.get("summary")
        source = item.get("source")

        ticker_sentiments = item.get("ticker_sentiment", [])
        matched = [
            s["ticker"]
            for s in ticker_sentiments
            if s.get("ticker") in ticker_set
        ]
        if not matched:
            if ticker_sentiments:
                # Article has ticker_sentiment entries but none are in our query set — skip.
                continue
            else:
                # ticker_sentiment absent/empty: AV returned this article for our query,
                # so attribute it to all queried tickers (common for single-ticker calls).
                matched = list(ticker_set)

        for ticker in matched:
            articles.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "url": url,
                }
            )

    return articles


def write_av_news(articles: list[dict], sector: str, engine: Engine) -> int:
    """Insert AV articles into news_raw, deduplicating on (url, ticker).

    Unlike the Finnhub writer which deduplicates by URL alone, this uses
    (url, ticker) pairs because the same article can legitimately cover
    multiple constituent tickers and should produce one row for each.

    Args:
        articles: Output of fetch_av_news.
        sector: ETF ticker for the sector (e.g. 'XLK').
        engine: SQLAlchemy engine.

    Returns:
        Number of new rows inserted.
    """
    if not articles:
        return 0

    urls = [a["url"] for a in articles if a.get("url")]

    with Session(engine) as session:
        existing_rows = session.execute(
            select(NewsRaw.url, NewsRaw.ticker).where(NewsRaw.url.in_(urls))
        ).all()
        existing_pairs: set[tuple[str, str]] = {(r.url, r.ticker) for r in existing_rows}

        new_rows = [
            NewsRaw(
                ticker=a["ticker"],
                sector=sector,
                timestamp=a["timestamp"],
                source=a.get("source"),
                title=a["title"],
                summary=a.get("summary"),
                url=a.get("url"),
            )
            for a in articles
            if a.get("url") and (a["url"], a["ticker"]) not in existing_pairs
        ]

        session.add_all(new_rows)
        session.commit()

    return len(new_rows)


def validate_av_historical_depth(
    ticker: str = "AAPL",
    months: int = 14,
    api_key: str | None = None,
) -> int:
    """Check how many months of news Alpha Vantage returns for a ticker.

    Args:
        ticker: Ticker to probe (default AAPL — large, well-covered stock).
        months: How far back to request (slightly over target to find true limit).
        api_key: Alpha Vantage API key.

    Returns:
        Approximate months of history available (floored integer).
    """
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=months * 31)
    articles = fetch_av_news([ticker], time_from=start, time_to=end, limit=200, api_key=api_key)

    if not articles:
        return 0

    earliest = min(a["timestamp"] for a in articles)
    return int((end - earliest).days / 30.44)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_av_timestamp(raw: str) -> datetime.datetime | None:
    """Parse Alpha Vantage time_published field (YYYYMMDDTHHMMSS or YYYYMMDDTHHMM)."""
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _rate_limit() -> None:
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < _CALL_INTERVAL:
        time.sleep(_CALL_INTERVAL - elapsed)
    _last_call_ts = time.monotonic()
