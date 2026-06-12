from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# load_dotenv before any local imports so FRED_API_KEY is available immediately
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import create_engine  # noqa: E402

from db.init import init_db  # noqa: E402
from ingestion.macro import SERIES_IDS, fetch_macro, write_macro  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
_LOOKBACK_MONTHS = 18


def main() -> None:
    end = datetime.date.today()
    start = (
        end.replace(year=end.year - 1)
        if end.month > 6
        else end.replace(year=end.year - 2, month=end.month + 6)
    )

    logger.info("Fetching %d FRED series: %s", len(SERIES_IDS), SERIES_IDS)
    logger.info("Date range: %s → %s", start, end)

    init_db(_DB_PATH)
    engine = create_engine(f"sqlite:///{_DB_PATH}")

    df = fetch_macro(SERIES_IDS, start, end)
    n = write_macro(df, engine)

    logger.info("Done — %d rows upserted across %d series", n, len(SERIES_IDS))
    engine.dispose()


if __name__ == "__main__":
    main()
