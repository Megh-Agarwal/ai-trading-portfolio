from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import create_engine  # noqa: E402

from db.init import init_db  # noqa: E402
from ingestion.polymarket import (  # noqa: E402
    fetch_current_state,
    load_curated_markets,
    write_polymarket,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"


def main() -> None:
    markets = load_curated_markets()
    logger.info("Loaded %d curated markets from YAML", len(markets))

    init_db(_DB_PATH)
    engine = create_engine(f"sqlite:///{_DB_PATH}")

    snapshots = fetch_current_state(markets)
    logger.info("Fetched %d snapshots from Gamma API", len(snapshots))

    n = write_polymarket(snapshots, engine)
    logger.info("Wrote %d rows to polymarket_raw", n)

    if len(snapshots) < len(markets):
        skipped = len(markets) - len(snapshots)
        logger.warning(
            "%d market(s) skipped — check that market IDs in "
            "config/polymarket_markets.yaml are still active on Polymarket.",
            skipped,
        )

    engine.dispose()


if __name__ == "__main__":
    main()
