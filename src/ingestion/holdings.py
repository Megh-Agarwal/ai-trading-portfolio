from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

_HOLDINGS_PATH = Path(__file__).parent.parent.parent / "config" / "sector_holdings.yaml"


def fetch_top_holdings(etf_ticker: str, n: int = 10) -> list[str]:
    """Fetch the top n holdings for an ETF via yfinance funds_data.

    Args:
        etf_ticker: ETF symbol (e.g. 'XLK').
        n: Number of top holdings to return (by weight, descending).

    Returns:
        List of constituent ticker symbols.
    """
    df: pd.DataFrame = yf.Ticker(etf_ticker).funds_data.top_holdings
    symbols = list(df.index[:n])
    logger.info("Fetched %d holdings for %s", len(symbols), etf_ticker)
    return symbols


def validate_holdings(holdings: dict[str, list[str]]) -> dict[str, list[str]]:
    """Identify tickers that appear in more than one sector ETF.

    Args:
        holdings: Mapping of ETF ticker → list of constituent tickers.

    Returns:
        Dict of constituent ticker → list of ETFs it appears in, for every
        ticker that appears in more than one sector. Empty means no overlap.
    """
    ticker_to_etfs: dict[str, list[str]] = defaultdict(list)
    for etf, tickers in holdings.items():
        for ticker in tickers:
            ticker_to_etfs[ticker].append(etf)
    return {t: etfs for t, etfs in ticker_to_etfs.items() if len(etfs) > 1}


def load_holdings(path: Path = _HOLDINGS_PATH) -> dict[str, list[str]]:
    """Load the cached holdings from config/sector_holdings.yaml.

    Args:
        path: Override the default config path (useful in tests).

    Returns:
        Mapping of ETF ticker → list of constituent tickers.

    Raises:
        FileNotFoundError: YAML cache does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Holdings cache not found: {path}. Run scripts/update_holdings.py first."
        )
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return {k: v for k, v in data.items() if isinstance(v, list)}
