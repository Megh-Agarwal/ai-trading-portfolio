from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from db.init import init_db  # noqa: E402
from db.models import NewsRaw  # noqa: E402
from ingestion.alpha_vantage_news import (  # noqa: E402
    _CALL_INTERVAL,
    fetch_av_news,
    validate_av_historical_depth,
    write_av_news,
)
from ingestion.holdings import load_holdings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
_BACKTEST_MONTHS = 18
_MIN_ACCEPTABLE_MONTHS = 10
_AV_FREE_LIMIT = 200
# AV free tier: 25 calls/day. Reserve 1 for depth validation → 24 for tickers.
_DAILY_TICKER_BUDGET = 24


def _tickers_needing_backfill(
    holdings: dict[str, list[str]],
    engine,
    start: datetime.datetime,
) -> list[tuple[str, str]]:
    """Return (ticker, etf) pairs that have no news articles on or after start.

    Already-ingested tickers are skipped so the script is safely resumable:
    re-run each day until all 100 tickers are done (4 days on the free tier).
    """
    cutoff = start.replace(hour=0, minute=0, second=0, microsecond=0)
    with Session(engine) as session:
        existing: set[str] = set(
            session.execute(
                select(NewsRaw.ticker.distinct()).where(NewsRaw.timestamp >= cutoff)
            ).scalars()
        )

    pending = []
    for etf, tickers in holdings.items():
        for ticker in tickers:
            if ticker not in existing:
                pending.append((ticker, etf))
    return pending


def main() -> None:
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=_BACKTEST_MONTHS * 31)

    # ------------------------------------------------------------------
    # Step 1: Validate historical depth
    # ------------------------------------------------------------------
    logger.info("Validating Alpha Vantage historical depth for AAPL...")
    depth = validate_av_historical_depth(ticker="AAPL", months=_BACKTEST_MONTHS + 2)
    logger.info("Alpha Vantage historical depth: ~%d months", depth)

    if depth < _MIN_ACCEPTABLE_MONTHS:
        logger.warning(
            "Alpha Vantage returned only ~%d months of history. "
            "Backtest window will be limited — check your API key tier.",
            depth,
        )
        start = end - datetime.timedelta(days=depth * 30)

    # ------------------------------------------------------------------
    # Step 2: One call per ticker, 24 tickers/day (free tier: 25 calls/day).
    # Script is resumable: tickers already in the DB are skipped automatically.
    # ------------------------------------------------------------------
    holdings = load_holdings()
    init_db(_DB_PATH)
    engine = create_engine(f"sqlite:///{_DB_PATH}")

    total_tickers = sum(len(v) for v in holdings.values())
    pending = _tickers_needing_backfill(holdings, engine, start)
    done_so_far = total_tickers - len(pending)

    if not pending:
        logger.info("All %d tickers already have news data — nothing to do.", total_tickers)
        engine.dispose()
        return

    batch = pending[:_DAILY_TICKER_BUDGET]
    logger.info(
        "Progress: %d/%d tickers done. Fetching %d today (daily budget = %d).",
        done_so_far, total_tickers, len(batch), _DAILY_TICKER_BUDGET,
    )

    total_articles = 0

    for ticker, etf in batch:
        try:
            articles = fetch_av_news(
                [ticker],
                time_from=start,
                time_to=end,
                limit=_AV_FREE_LIMIT,
            )

            if len(articles) >= _AV_FREE_LIMIT:
                logger.warning(
                    "%s: returned %d articles (at limit) — some may be truncated. "
                    "Consider an AV premium plan for limit=1000.",
                    ticker, len(articles),
                )

            n = write_av_news(articles, sector=etf, engine=engine)
            total_articles += n
            logger.info("%s (%s): %d new articles ingested", ticker, etf, n)

        except Exception as exc:
            logger.error("Failed %s: %s", ticker, exc)

    remaining = len(pending) - len(batch)
    est_minutes = (len(batch) * _CALL_INTERVAL) / 60
    logger.info(
        "Done — %d new articles, %d tickers fetched (~%.1f min wall time).",
        total_articles, len(batch), est_minutes,
    )

    if remaining > 0:
        logger.info(
            "%d tickers remaining — re-run tomorrow to continue "
            "(full backfill takes %d days on the free tier).",
            remaining,
            -(-total_tickers // _DAILY_TICKER_BUDGET),  # ceiling division
        )
    else:
        logger.info("All %d tickers fully ingested!", total_tickers)

    engine.dispose()


if __name__ == "__main__":
    main()
