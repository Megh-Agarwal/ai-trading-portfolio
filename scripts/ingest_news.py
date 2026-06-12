from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import create_engine  # noqa: E402

from db.init import init_db  # noqa: E402
from ingestion.holdings import load_holdings  # noqa: E402
from ingestion.news import (  # noqa: E402
    _CALL_INTERVAL,
    fetch_company_news,
    validate_historical_depth,
    write_news,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
_BACKTEST_MONTHS = 12
_MIN_ACCEPTABLE_MONTHS = 10  # flag if Finnhub returns less than this


def _month_ranges(
    start: datetime.date, end: datetime.date
) -> list[tuple[datetime.date, datetime.date]]:
    """Split [start, end] into consecutive calendar-month chunks."""
    ranges = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end_day = (cursor.replace(month=cursor.month % 12 + 1, day=1) if cursor.month < 12
                         else cursor.replace(year=cursor.year + 1, month=1, day=1))
        chunk_end = min(month_end_day - datetime.timedelta(days=1), end)
        ranges.append((max(cursor, start), chunk_end))
        cursor = month_end_day
    return ranges


def main() -> None:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=_BACKTEST_MONTHS * 31)

    # -------------------------------------------------------------------
    # Step 1: Validate historical depth before committing to full backfill
    # -------------------------------------------------------------------
    logger.info("Validating Finnhub historical depth for AAPL...")
    depth = validate_historical_depth(ticker="AAPL", months=14)
    logger.info("Finnhub historical depth: ~%d months", depth)

    if depth < _MIN_ACCEPTABLE_MONTHS:
        logger.warning(
            "Finnhub free tier only returns ~%d months of history. "
            "Backtest window will be limited — see decisions.md.",
            depth,
        )
        start = end - datetime.timedelta(days=depth * 30)

    # -------------------------------------------------------------------
    # Step 2: Full backfill — loop sectors → tickers → monthly chunks
    # -------------------------------------------------------------------
    holdings = load_holdings()
    init_db(_DB_PATH)
    engine = create_engine(f"sqlite:///{_DB_PATH}")

    month_chunks = _month_ranges(start, end)
    total_articles = 0
    total_calls = 0

    for etf, tickers in holdings.items():
        etf_articles = 0
        for ticker in tickers:
            for chunk_start, chunk_end in month_chunks:
                try:
                    articles = fetch_company_news(ticker, chunk_start, chunk_end)
                    n = write_news(articles, sector=etf, engine=engine)
                    etf_articles += n
                    total_articles += n
                    total_calls += 1
                    logger.debug(
                        "%s [%s→%s]: %d articles (%d new)",
                        ticker, chunk_start, chunk_end, len(articles), n,
                    )
                except Exception as exc:
                    logger.error("Failed %s %s→%s: %s", ticker, chunk_start, chunk_end, exc)
        logger.info("%s: %d new articles ingested", etf, etf_articles)

    est_minutes = (total_calls * _CALL_INTERVAL) / 60
    logger.info(
        "Done — %d new articles from %d API calls (~%.0f min wall time)",
        total_articles, total_calls, est_minutes,
    )
    engine.dispose()


if __name__ == "__main__":
    main()
