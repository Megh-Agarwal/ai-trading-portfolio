from __future__ import annotations

import datetime
import logging
import os
import time

import finnhub
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from db.models import NewsRaw

logger = logging.getLogger(__name__)

# Finnhub free tier: 60 calls/min. 1.1s gives comfortable headroom.
_CALL_INTERVAL = 1.1
_last_call_ts: float = 0.0


def fetch_company_news(
    ticker: str,
    start: datetime.date,
    end: datetime.date,
    api_key: str | None = None,
) -> list[dict]:
    """Fetch company news from Finnhub for a single ticker and date range.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        start: Inclusive start date.
        end: Inclusive end date.
        api_key: Finnhub API key. Defaults to FINNHUB_API_KEY env var.

    Returns:
        List of normalized article dicts with keys:
        ticker, timestamp, source, title, summary, url.
    """
    key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise EnvironmentError("FINNHUB_API_KEY not set. Add it to .env or export it.")

    _rate_limit()
    client = finnhub.Client(api_key=key)
    raw = client.company_news(ticker, _from=start.isoformat(), to=end.isoformat()) or []

    articles = []
    for item in raw:
        ts = item.get("datetime")
        if not ts:
            continue
        articles.append(
            {
                "ticker": ticker,
                "timestamp": datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).replace(
                    tzinfo=None
                ),
                "source": item.get("source"),
                "title": item.get("headline") or "",
                "summary": item.get("summary"),
                "url": item.get("url"),
            }
        )
    return articles


def write_news(articles: list[dict], sector: str, engine: Engine) -> int:
    """Insert new articles into news_raw, deduplicating by URL.

    Args:
        articles: Output of fetch_company_news.
        sector: ETF ticker for the sector (e.g. 'XLK').
        engine: SQLAlchemy engine.

    Returns:
        Number of new rows inserted (skips articles with existing URLs).
    """
    if not articles:
        return 0

    urls = [a["url"] for a in articles if a.get("url")]

    with Session(engine) as session:
        existing_urls: set[str] = set(
            session.execute(select(NewsRaw.url).where(NewsRaw.url.in_(urls))).scalars()
        )

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
            if a.get("url") not in existing_urls
        ]

        session.add_all(new_rows)
        session.commit()

    return len(new_rows)


def aggregate_to_sector_week(
    etf: str,
    week_start: datetime.date,
    engine: Engine,
    holdings: dict[str, list[str]] | None = None,
) -> dict:
    """Aggregate news from the DB for a sector ETF over a calendar week.

    Args:
        etf: Sector ETF ticker (e.g. 'XLK').
        week_start: Monday of the target week.
        engine: SQLAlchemy engine.
        holdings: Optional pre-loaded holdings map. Loads from YAML if None.

    Returns:
        Dict with keys: etf, week_start, week_end, article_count,
        unique_url_count, tickers_covered, articles.
    """
    if holdings is None:
        from ingestion.holdings import load_holdings

        holdings = load_holdings()

    tickers = holdings.get(etf, [])
    week_end = week_start + datetime.timedelta(days=6)
    week_start_dt = datetime.datetime.combine(week_start, datetime.time.min)
    week_end_dt = datetime.datetime.combine(week_end, datetime.time.max)

    with Session(engine) as session:
        rows = (
            session.execute(
                select(NewsRaw)
                .where(NewsRaw.ticker.in_(tickers))
                .where(NewsRaw.timestamp >= week_start_dt)
                .where(NewsRaw.timestamp <= week_end_dt)
                .order_by(NewsRaw.timestamp.desc())
            )
            .scalars()
            .all()
        )

    seen_urls: set[str] = set()
    articles = []
    for row in rows:
        if row.url and row.url in seen_urls:
            continue
        if row.url:
            seen_urls.add(row.url)
        articles.append(
            {
                "ticker": row.ticker,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "source": row.source,
                "title": row.title,
                "url": row.url,
            }
        )

    tickers_covered = sorted({a["ticker"] for a in articles if a["ticker"]})

    return {
        "etf": etf,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "article_count": len(articles),
        "unique_url_count": len(seen_urls),
        "tickers_covered": tickers_covered,
        "articles": articles,
    }


def validate_historical_depth(
    ticker: str = "AAPL",
    months: int = 14,
    api_key: str | None = None,
) -> int:
    """Check how many months of news Finnhub returns for a ticker.

    Args:
        ticker: Ticker to probe (default AAPL — large, well-covered stock).
        months: How far back to request (slightly over target to find true limit).
        api_key: Finnhub API key.

    Returns:
        Approximate months of history available (floored integer).
    """
    key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise EnvironmentError("FINNHUB_API_KEY not set.")

    _rate_limit()
    client = finnhub.Client(api_key=key)
    end = datetime.date.today()
    start = end - datetime.timedelta(days=months * 31)
    raw = client.company_news(ticker, _from=start.isoformat(), to=end.isoformat()) or []

    if not raw:
        return 0

    earliest_ts = min(item["datetime"] for item in raw)
    earliest_date = datetime.datetime.fromtimestamp(earliest_ts).date()
    return int((end - earliest_date).days / 30.44)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _rate_limit() -> None:
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < _CALL_INTERVAL:
        time.sleep(_CALL_INTERVAL - elapsed)
    _last_call_ts = time.monotonic()
