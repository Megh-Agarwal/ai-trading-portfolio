from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from config import load_config
from ingestion.holdings import fetch_top_holdings, validate_holdings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HOLDINGS_PATH = Path(__file__).parent.parent / "config" / "sector_holdings.yaml"
_N = 10


def main() -> None:
    universe = load_config("universe")
    etf_tickers = universe.ticker_list

    logger.info("Fetching top %d holdings for %d ETFs", _N, len(etf_tickers))

    holdings: dict[str, list[str]] = {}
    for etf in etf_tickers:
        holdings[etf] = fetch_top_holdings(etf, n=_N)

    overlaps = validate_holdings(holdings)
    if overlaps:
        logger.warning("Cross-sector overlaps detected (see ADR-007): %s", overlaps)
    else:
        logger.info("Validation passed — no cross-sector ticker overlaps")

    total = sum(len(v) for v in holdings.values())
    logger.info("%d total holdings across %d ETFs", total, len(holdings))

    overlap_note = str(overlaps) if overlaps else "none"
    header = (
        f"# Sector ETF top-{_N} holdings cache — refresh quarterly\n"
        f"# Generated: {datetime.date.today().isoformat()} via yfinance funds_data.top_holdings\n"
        f"# Cross-sector overlaps: {overlap_note}\n"
        "#\n"
    )

    with _HOLDINGS_PATH.open("w") as fh:
        fh.write(header)
        yaml.dump(holdings, fh, default_flow_style=False, sort_keys=True)

    logger.info("Written to %s", _HOLDINGS_PATH)


if __name__ == "__main__":
    main()
