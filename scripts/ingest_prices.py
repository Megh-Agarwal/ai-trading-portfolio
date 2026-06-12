from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine

from config import load_config
from db.init import init_db
from ingestion.prices import fetch_prices, write_prices

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
_LOOKBACK_MONTHS = 18


def main() -> None:
    universe = load_config("universe")
    tickers = [t.ticker for t in universe.tickers] + [universe.benchmark]

    end = datetime.date.today()
    # Approximate 18 months by subtracting days (avoids dateutil dependency)
    start = (end.replace(year=end.year - 1) if end.month > 6
             else end.replace(year=end.year - 2, month=end.month + 6))

    logger.info("Universe: %s", tickers)
    logger.info("Date range: %s → %s (%d months)", start, end, _LOOKBACK_MONTHS)

    init_db(_DB_PATH)
    engine = create_engine(f"sqlite:///{_DB_PATH}")

    df = fetch_prices(tickers, start, end)
    n = write_prices(df, engine)

    logger.info("Done — %d rows upserted across %d tickers", n, len(tickers))
    engine.dispose()


if __name__ == "__main__":
    main()
