from __future__ import annotations

import datetime
import logging

import pandas as pd
import yfinance as yf
from sqlalchemy import Engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from db.models import Price

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"open", "high", "low", "close", "volume", "adj_close"}

_COL_RENAME = {
    "Adj Close": "adj_close",
    "Close": "close",
    "High": "high",
    "Low": "low",
    "Open": "open",
    "Volume": "volume",
}


def fetch_prices(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    """Download OHLCV from yfinance and return a validated long-format DataFrame.

    Args:
        tickers: List of ticker symbols.
        start: Inclusive start date.
        end: Exclusive end date (yfinance convention).

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume, adj_close.
        Rows with missing close prices are dropped after quality checks are logged.
    """
    logger.info("Downloading %d tickers %s → %s", len(tickers), start, end)

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.warning("yfinance returned empty DataFrame for %s", tickers)
        return pd.DataFrame(columns=["date", "ticker"] + sorted(_REQUIRED_COLS))

    df = _reshape(raw, tickers)
    _check_quality(df)

    before = len(df)
    df = df.dropna(subset=["close"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN close — see quality warnings above", dropped)

    return df.reset_index(drop=True)


def write_prices(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert price rows into the prices table (idempotent).

    Args:
        df: Long-format DataFrame from fetch_prices.
        engine: SQLAlchemy engine.

    Returns:
        Number of rows processed.
    """
    if df.empty:
        return 0

    rows = [
        {
            "date": row.date,
            "ticker": row.ticker,
            "open": float(row.open) if pd.notna(row.open) else None,
            "high": float(row.high) if pd.notna(row.high) else None,
            "low": float(row.low) if pd.notna(row.low) else None,
            "close": float(row.close),
            "volume": int(row.volume) if pd.notna(row.volume) else 0,
            "adj_close": float(row.adj_close) if pd.notna(row.adj_close) else None,
        }
        for row in df.itertuples(index=False)
    ]

    stmt = sqlite_insert(Price).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date", "ticker"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "adj_close": stmt.excluded.adj_close,
        },
    )

    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

    logger.info("Upserted %d price rows", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _reshape(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        ticker = tickers[0]
        raw = raw.copy()
        raw.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in raw.columns])

    long = raw.stack(level=1)  # → (Date, Ticker) index, field columns
    long.index.names = ["date", "ticker"]
    long = long.rename(columns=_COL_RENAME)

    ordered = ["open", "high", "low", "close", "volume", "adj_close"]
    present = [c for c in ordered if c in long.columns]
    long = long[present].reset_index()

    long["date"] = pd.to_datetime(long["date"]).dt.date
    return long


def _check_quality(df: pd.DataFrame) -> None:
    for ticker, grp in df.groupby("ticker"):
        nan_close = int(grp["close"].isna().sum())
        if nan_close:
            logger.warning("QUALITY [%s]: %d NaN close prices", ticker, nan_close)

        if "close" in grp.columns:
            bad = grp[grp["close"].notna() & (grp["close"] <= 0)]
            if not bad.empty:
                logger.warning("QUALITY [%s]: %d zero/negative close prices on %s",
                               ticker, len(bad), bad["date"].tolist())

        if "adj_close" in grp.columns:
            bad_adj = grp[grp["adj_close"].notna() & (grp["adj_close"] <= 0)]
            if not bad_adj.empty:
                logger.warning("QUALITY [%s]: %d zero/negative adj_close values",
                               ticker, len(bad_adj))

            valid = grp[grp["close"].notna() & grp["adj_close"].notna() & (grp["adj_close"] > 0)]
            if not valid.empty:
                ratio = valid["close"] / valid["adj_close"]
                suspicious = valid[(ratio > 20) | (ratio < 0.05)]
                if not suspicious.empty:
                    logger.warning(
                        "QUALITY [%s]: %d rows with suspicious close/adj_close ratio (%.2f–%.2f)",
                        ticker, len(suspicious), ratio.min(), ratio.max(),
                    )
