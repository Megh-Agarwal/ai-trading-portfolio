from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd
import requests
import yaml
from sqlalchemy import Engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from db.models import PolymarketRaw

logger = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_TIMEOUT = 15  # seconds per request
_CURATED_PATH = Path(__file__).parent.parent.parent / "config" / "polymarket_markets.yaml"


def fetch_active_markets(category: str = "economy", limit: int = 100) -> list[dict]:
    """Fetch active (unresolved) markets from the Gamma API.

    Args:
        category: Tag slug to filter by (e.g. 'economy', 'politics').
        limit: Maximum number of markets to return per request.

    Returns:
        List of raw market dicts from the Gamma API. Each dict includes at
        minimum: conditionId, question, outcomePrices, volume, endDate.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
        requests.RequestException: On network failure.
    """
    params: dict = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "tag": category,
    }
    resp = requests.get(f"{_GAMMA_BASE}/markets", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_market_prices(
    condition_id: str,
    start: datetime.date,
    end: datetime.date,
    fidelity: int = 1440,
) -> pd.DataFrame:
    """Fetch daily implied-probability history for a binary market.

    Queries the CLOB prices-history endpoint. The YES token price is treated
    as the implied probability (0–1 range).

    Args:
        condition_id: Polymarket condition ID (hex string from Gamma API).
        start: Inclusive start date.
        end: Inclusive end date.
        fidelity: Candle size in minutes (default 1440 = one candle per day).

    Returns:
        DataFrame with columns timestamp (datetime, UTC-naive) and price (float 0-1).
        Returns an empty DataFrame when no data is available or the API fails.
    """
    start_ts = int(datetime.datetime.combine(start, datetime.time.min).timestamp())
    end_ts = int(datetime.datetime.combine(end, datetime.time.max).timestamp())
    params: dict = {
        "market": condition_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }
    try:
        resp = requests.get(f"{_CLOB_BASE}/prices-history", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        history = resp.json().get("history", [])
    except requests.RequestException as exc:
        logger.warning("prices-history failed for %s: %s", condition_id, exc)
        return pd.DataFrame(columns=["timestamp", "price"])

    if not history:
        return pd.DataFrame(columns=["timestamp", "price"])

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"p": "price"})[["timestamp", "price"]]
    df["price"] = df["price"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_curated_markets(path: Path | None = None) -> list[dict]:
    """Load the curated macro-relevant market list from YAML.

    Args:
        path: Path to YAML file. Defaults to config/polymarket_markets.yaml.

    Returns:
        List of market dicts with keys: market_id, question, sector_impacts,
        confidence, category.
    """
    yaml_path = path or _CURATED_PATH
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    return data.get("markets", [])


def fetch_current_state(markets: list[dict]) -> list[dict]:
    """Fetch the current implied probability snapshot for each curated market.

    Calls the Gamma API once per market. Markets that return a non-2xx
    response (e.g. expired/delisted) are skipped with a warning.

    Args:
        markets: Output of load_curated_markets().

    Returns:
        List of snapshot dicts ready to pass to write_polymarket().
    """
    now = datetime.datetime.utcnow().replace(microsecond=0)
    snapshots = []
    for m in markets:
        mid = m["market_id"]
        try:
            resp = requests.get(f"{_GAMMA_BASE}/markets/{mid}", timeout=_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException as exc:
            logger.warning("Skipping market %s — API error: %s", mid, exc)
            continue

        yes_prob = _extract_yes_prob(raw)
        snapshots.append(
            {
                "market_id": mid,
                "timestamp": now,
                "question": raw.get("question") or m.get("question", ""),
                "implied_prob": yes_prob,
                "volume": float(raw.get("volume") or 0),
                "category": raw.get("category") or m.get("category"),
                "end_date": _parse_end_date(raw.get("endDate")),
            }
        )
        logger.debug("%s  p(YES)=%.3f  vol=%.0f", mid, yes_prob, float(raw.get("volume") or 0))
    return snapshots


def write_polymarket(snapshots: list[dict], engine: Engine) -> int:
    """Upsert market snapshots into polymarket_raw, keyed on (market_id, timestamp).

    Args:
        snapshots: Output of fetch_current_state().
        engine: SQLAlchemy engine.

    Returns:
        Number of rows inserted or updated.
    """
    if not snapshots:
        return 0

    rows = [
        {
            "market_id": s["market_id"],
            "timestamp": s["timestamp"],
            "question": s["question"],
            "implied_prob": s["implied_prob"],
            "volume": s.get("volume"),
            "category": s.get("category"),
            "end_date": s.get("end_date"),
        }
        for s in snapshots
    ]

    with Session(engine) as session:
        stmt = sqlite_insert(PolymarketRaw).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["market_id", "timestamp"],
            set_={
                "implied_prob": stmt.excluded.implied_prob,
                "volume": stmt.excluded.volume,
                "category": stmt.excluded.category,
                "end_date": stmt.excluded.end_date,
            },
        )
        session.execute(stmt)
        session.commit()

    return len(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_yes_prob(raw: dict) -> float:
    """Parse the YES implied probability from a Gamma API market dict."""
    outcome_prices = raw.get("outcomePrices")
    if isinstance(outcome_prices, list) and outcome_prices:
        try:
            return float(outcome_prices[0])
        except (ValueError, TypeError):
            pass
    # Fall back to tokens array
    for token in raw.get("tokens", []):
        if token.get("outcome", "").upper() == "YES":
            try:
                return float(token.get("price", float("nan")))
            except (ValueError, TypeError):
                pass
    return float("nan")


def _parse_end_date(raw: str | None) -> datetime.date | None:
    """Parse ISO-8601 date strings returned by the Gamma API."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None
