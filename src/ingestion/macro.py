from __future__ import annotations

import datetime
import logging
import os

import pandas as pd
from fredapi import Fred
from sqlalchemy import Engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from db.models import Macro

logger = logging.getLogger(__name__)

# Fetch this many days before `start` so that monthly/weekly series have a
# value to forward-fill from on the first day of the requested window.
_BUFFER_DAYS = 90

SERIES_IDS = [
    "T10Y2Y",  # 10Y-2Y Treasury spread (daily)
    "DGS10",  # 10Y Treasury yield (daily)
    "VIXCLS",  # VIX (daily)
    "DTWEXBGS",  # Trade-weighted USD index (daily)
    "CPIAUCSL",  # CPI (monthly)
    "UNRATE",  # Unemployment rate (monthly)
    "ICSA",  # Initial jobless claims (weekly)
]


def fetch_macro(
    series_ids: list[str],
    start: datetime.date,
    end: datetime.date,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download macro series from FRED and forward-fill to daily granularity.

    Monthly and weekly series (CPI, UNRATE, ICSA) are reindexed to a daily
    date range and forward-filled — the most recently released value is
    propagated forward. A 90-day buffer before `start` is fetched to ensure
    the first day of the window always has a value.

    Args:
        series_ids: FRED series IDs to fetch.
        start: Inclusive start date for the returned DataFrame.
        end: Inclusive end date for the returned DataFrame.
        api_key: FRED API key. Defaults to FRED_API_KEY env var.

    Returns:
        Long-format DataFrame with columns: date, series_id, value.
        No NaN values within [start, end] — rows with unavoidable NaN are
        dropped with a warning.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise EnvironmentError("FRED_API_KEY not set. Add it to .env or export it.")

    fred = Fred(api_key=key)
    fetch_start = start - datetime.timedelta(days=_BUFFER_DAYS)

    dfs: list[pd.DataFrame] = []
    for sid in series_ids:
        raw = _fetch_series(fred, sid, fetch_start, end)
        if raw is None or raw.empty:
            logger.warning("No data returned for %s — skipping", sid)
            continue

        # Reindex over buffer + window, forward-fill, then clip to [start, end]
        full_index = pd.date_range(pd.Timestamp(fetch_start), pd.Timestamp(end), freq="D")
        daily = raw.reindex(full_index).ffill()
        daily = daily.loc[pd.Timestamp(start) : pd.Timestamp(end)]

        nan_count = int(daily.isna().sum())
        if nan_count:
            logger.warning(
                "QUALITY [%s]: %d NaN values after forward-fill "
                "(series may start after requested window) — rows dropped",
                sid,
                nan_count,
            )
            daily = daily.dropna()

        df_s = daily.reset_index()
        df_s.columns = pd.Index(["date", "value"])
        df_s["series_id"] = sid
        df_s["date"] = pd.to_datetime(df_s["date"]).dt.date
        dfs.append(df_s[["date", "series_id", "value"]])
        logger.info("Fetched %s: %d daily rows", sid, len(df_s))

    if not dfs:
        return pd.DataFrame(columns=["date", "series_id", "value"])

    result = pd.concat(dfs, ignore_index=True)
    _check_quality(result, start, end)
    return result


def write_macro(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert macro rows into the macro table (idempotent).

    Args:
        df: Long-format DataFrame from fetch_macro.
        engine: SQLAlchemy engine.

    Returns:
        Number of rows processed.
    """
    if df.empty:
        return 0

    rows = [
        {"date": row.date, "series_id": row.series_id, "value": float(row.value)}
        for row in df.itertuples(index=False)
    ]

    stmt = sqlite_insert(Macro).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date", "series_id"],
        set_={"value": stmt.excluded.value},
    )

    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

    logger.info("Upserted %d macro rows", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_series(
    fred: Fred,
    series_id: str,
    start: datetime.date,
    end: datetime.date,
) -> pd.Series | None:
    try:
        raw = fred.get_series(
            series_id,
            observation_start=start.isoformat(),
            observation_end=end.isoformat(),
        )
        return raw
    except Exception as exc:
        logger.error("FRED request failed for %s: %s", series_id, exc)
        raise


def _check_quality(df: pd.DataFrame, start: datetime.date, end: datetime.date) -> None:
    window = df[(df["date"] >= start) & (df["date"] <= end)]
    for sid, grp in window.groupby("series_id"):
        nan_count = int(grp["value"].isna().sum())
        if nan_count:
            logger.warning("QUALITY [%s]: %d NaN values within backtest window", sid, nan_count)
        zero_neg = grp[grp["value"] <= 0]
        if not zero_neg.empty:
            logger.warning("QUALITY [%s]: %d zero/negative values", sid, len(zero_neg))
