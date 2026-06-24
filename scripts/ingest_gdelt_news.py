"""Ingest historical news from GDELT BigQuery into news_raw.

Backtest window: 2025-06-06 → 2026-06-12 (locked).
Queries one calendar month per ticker per BigQuery call for partition pruning.

Usage:
  # 3-ticker sanity check (run this first)
  uv run python scripts/ingest_gdelt_news.py --tickers NVDA JPM NEE

  # Full 100-ticker backfill (run after sanity check passes)
  uv run python scripts/ingest_gdelt_news.py

  # Resume after interruption (skips tickers already in checkpoint file)
  uv run python scripts/ingest_gdelt_news.py --resume
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine  # noqa: E402

from db.init import init_db  # noqa: E402
from ingestion.gdelt import fetch_gdelt_news, load_company_names, write_gdelt_news  # noqa: E402
from ingestion.holdings import load_holdings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
_CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "gdelt_checkpoint.json"
# Locked backtest window — matches the range established by macro/price data
_DATE_FROM = datetime.date(2025, 6, 6)
_DATE_TO = datetime.date(2026, 6, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GDELT news into news_raw")
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Restrict to specific tickers (e.g. --tickers NVDA JPM NEE). "
             "Omit to run all 100 constituents.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers already recorded in data/gdelt_checkpoint.json.",
    )
    args = parser.parse_args()

    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        logger.error("GCP_PROJECT_ID not set in environment / .env file")
        sys.exit(1)

    company_names_map = load_company_names()
    holdings = load_holdings()

    # Build ETF lookup: ticker → sector ETF
    ticker_to_etf: dict[str, str] = {}
    for etf, tickers in holdings.items():
        for t in tickers:
            ticker_to_etf[t] = etf

    # Determine which tickers to run
    if args.tickers:
        requested = [t.upper() for t in args.tickers]
        missing_names = [t for t in requested if t not in company_names_map]
        missing_etf = [t for t in requested if t not in ticker_to_etf]
        if missing_names:
            logger.error(
                "Tickers not in ticker_company_names.yaml: %s", missing_names
            )
            sys.exit(1)
        if missing_etf:
            logger.error(
                "Tickers not in sector_holdings.yaml: %s", missing_etf
            )
            sys.exit(1)
        work: list[tuple[str, str]] = [
            (t, ticker_to_etf[t]) for t in requested
        ]
    else:
        work = [
            (ticker, etf)
            for etf, tickers in holdings.items()
            for ticker in tickers
            if ticker in company_names_map
        ]
        missing = [
            ticker
            for etf, tickers in holdings.items()
            for ticker in tickers
            if ticker not in company_names_map
        ]
        if missing:
            logger.warning(
                "Skipping %d tickers with no company name mapping: %s",
                len(missing),
                missing,
            )

    # Load checkpoint — set of tickers already successfully completed
    completed: set[str] = set()
    if args.resume and _CHECKPOINT_PATH.exists():
        completed = set(json.loads(_CHECKPOINT_PATH.read_text()).get("completed", []))
        logger.info("Resuming: %d tickers already done, skipping them", len(completed))

    if args.resume:
        work = [(t, etf) for t, etf in work if t not in completed]
        if not work:
            logger.info("All tickers already completed — nothing to do.")
            return

    init_db(_DB_PATH)
    engine = create_engine(f"sqlite:///{_DB_PATH}")

    logger.info(
        "Starting GDELT backfill: %d tickers, %s → %s",
        len(work),
        _DATE_FROM,
        _DATE_TO,
    )

    grand_total_bytes = 0
    grand_total_rows = 0

    for i, (ticker, etf) in enumerate(work, 1):
        names = company_names_map[ticker]
        logger.info(
            "[%d/%d] %s (%s) — names: %s",
            i,
            len(work),
            ticker,
            etf,
            names,
        )
        try:
            articles, bytes_scanned = fetch_gdelt_news(
                ticker=ticker,
                company_names=names,
                date_from=_DATE_FROM,
                date_to=_DATE_TO,
                project_id=project_id,
            )
            rows_written = write_gdelt_news(articles, sector=etf, engine=engine)
            grand_total_bytes += bytes_scanned
            grand_total_rows += rows_written
            logger.info(
                "  → %d articles fetched, %d rows written, %.1f GB scanned this ticker",
                len(articles),
                rows_written,
                bytes_scanned / 1e9,
            )
            # Checkpoint: record this ticker as done so --resume can skip it
            completed.add(ticker)
            _CHECKPOINT_PATH.write_text(json.dumps({"completed": sorted(completed)}))
        except Exception:
            logger.exception("Failed %s — skipping", ticker)

    logger.info(
        "Done. Total rows written: %d | Total bytes scanned: %.2f GB",
        grand_total_rows,
        grand_total_bytes / 1e9,
    )
    engine.dispose()


if __name__ == "__main__":
    main()
