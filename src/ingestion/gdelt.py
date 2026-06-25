"""GDELT BigQuery ingestion for historical news sentiment.

Queries the GDELT 2.0 Global Knowledge Graph (GKG) public dataset on BigQuery.
One query per (ticker, calendar month) for partition pruning — keeps individual
query sizes to ~3–4 GB scanned vs ~560 GB on the unpartitioned gkg table.

Schema note: GDELT GKG has no article titles or body text, only URLs, source
names, organizations, and pre-computed tone scores. These are stored in the
existing news_raw schema as follows:
  title   → "[GDELT/{source}] {ticker}: {tone_direction} coverage (tone {n:+.1f})"
  summary → "tone {n:+.2f} | positive {p:.1f}% | negative {n:.1f}% | {w} words"
The news agent's LLM receives these fields and interprets tone scores as
pre-computed sentiment evidence.

Relevance filter (applied at write-time, before DB insert):
  Tier 1 (score ×3): top-tier financial/general press (Bloomberg, Reuters, etc.)
  Tier 2 (score ×2): solid financial/tech media (CNBC, Yahoo Finance, TechCrunch, etc.)
  Wire services (prnewswire.com, businesswire.com, globenewswire.com): excluded
    entirely — fetching the Themes column to gate them triples scan cost (~2.9→7.4
    GB/month). Editorial Tier 1+2 sources already cover market-relevant announcements.
  Tier 3 (score ×1): fallback only when a ticker-week has <5 Tier 1+2 articles.
  Everything else: discarded.

Combined score = tier × 2 + min(word_count/1000, 1.0) + min(|tone|/5.0, 1.0).
Top 50 per ticker per ISO week are kept; the rest are discarded before DB write.
URLs are normalised (strip scheme + www.) before dedup to collapse http/https pairs.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections import defaultdict
from pathlib import Path

import yaml
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from db.models import NewsRaw

logger = logging.getLogger(__name__)

_GDELT_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
_MIN_WORD_COUNT = 50
_TOP_N_PER_WEEK = 50
_TIER3_FALLBACK_THRESHOLD = 5  # use Tier 3 only if Tier 1+2 yield fewer than this
_NAMES_PATH = Path(__file__).parent.parent.parent / "config" / "ticker_company_names.yaml"
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 &.,'\-/]+$")

# Source allowlists — lowercase domain only (no scheme, no www.)
_TIER1: frozenset[str] = frozenset(
    {
        "reuters.com",
        "bloomberg.com",
        "wsj.com",
        "ft.com",
        "apnews.com",
        "nytimes.com",
        "washingtonpost.com",
        "economist.com",
        "barrons.com",
    }
)
_TIER2: frozenset[str] = frozenset(
    {
        "cnbc.com",
        "marketwatch.com",
        "fool.com",
        "benzinga.com",
        "seekingalpha.com",
        "forbes.com",
        "fortune.com",
        "businessinsider.com",
        "thestreet.com",
        "yahoo.com",
        "morningstar.com",
        "marketscreener.com",
        "investing.com",
        "insidermonkey.com",
        "techcrunch.com",
        "wired.com",
        "zdnet.com",
        "cnet.com",
        "theverge.com",
        "technologyreview.com",
        "arstechnica.com",
        "venturebeat.com",
        "utilitydive.com",
        "eenews.net",
        "electrek.co",
        "foxbusiness.com",
        "cnn.com",
        "nbcnews.com",
        "cbsnews.com",
        "npr.org",
        "theguardian.com",
        "abcnews.com",
        "bnnbloomberg.ca",
        "oilprice.com",
        "pv-tech.org",
        "cleantechnica.com",
        "investorplace.com",
        "investors.com",
    }
)
# Wire services excluded from GDELT backfill — press releases are noise without
# the Themes column (which triples scan cost). Editorial sources already cover
# market-relevant wire announcements. Wire services remain available via Finnhub
# for the weekly live refresh.
_WIRE: frozenset[str] = frozenset(
    {
        "prnewswire.com",
        "businesswire.com",
        "globenewswire.com",
    }
)
# Tier 3: used only as fallback when Tier 1+2 < _TIER3_FALLBACK_THRESHOLD per week
_TIER3: frozenset[str] = frozenset(
    {
        "financialcontent.com",
        "aol.com",
        "msn.com",
        "tomshardware.com",
        "wccftech.com",
        "webpronews.com",
        "defenseworld.net",
        "livemint.com",
        "moneycontrol.com",
        "businesstimes.com.sg",
        "thegazette.com",
        "briefingwire.com",
        "pr-inside.com",
        "techradar.com",
        "rttnews.com",
        "proactiveinvestors.com",
        "pv-magazine.com",
        "renewableenergyworld.com",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_company_names(path: Path = _NAMES_PATH) -> dict[str, list[str]]:
    """Load ticker→company-names mapping from config/ticker_company_names.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}


def fetch_gdelt_news(
    ticker: str,
    company_names: list[str],
    date_from: datetime.date,
    date_to: datetime.date,
    project_id: str,
) -> tuple[list[dict], int]:
    """Fetch GDELT GKG articles mentioning a company across a date range.

    Queries BigQuery in monthly chunks so each query prunes to one month of
    GDELT partitions (~3–4 GB scanned per ticker per month).

    Args:
        ticker: Constituent ticker (e.g. 'NVDA') — stored on the returned dicts.
        company_names: Names to match in Organizations / V2Organizations fields.
            Primary name first; alternates (rebrandings, spellings) follow.
        date_from: First date to include (inclusive).
        date_to: Last date to include (inclusive).
        project_id: GCP project ID for billing. Reads credentials from
            GOOGLE_APPLICATION_CREDENTIALS env var automatically.

    Returns:
        Tuple of (articles, total_bytes_processed).
        Each article dict has keys: ticker, timestamp, source, url, themes,
        tone, positive_score, negative_score, word_count.
    """
    from google.cloud import bigquery  # lazy import — not all callers need BQ

    _validate_names(company_names)

    client = bigquery.Client(project=project_id)
    all_articles: list[dict] = []
    total_bytes = 0

    for month_start, month_end in _month_chunks(date_from, date_to):
        sql = _build_query(company_names, month_start, month_end)
        job = client.query(sql)
        rows = list(job.result())
        bytes_this = job.total_bytes_processed or 0
        total_bytes += bytes_this
        logger.debug(
            "GDELT %s %s–%s: %d rows, %.1f MB scanned",
            ticker,
            month_start,
            month_end,
            len(rows),
            bytes_this / 1e6,
        )

        for row in rows:
            if not row.url:
                continue
            all_articles.append(
                {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.combine(row.article_date, datetime.time.min),
                    "source": row.source,
                    "url": row.url,
                    "tone": row.tone if row.tone is not None else 0.0,
                    "positive_score": (
                        row.positive_score if row.positive_score is not None else 0.0
                    ),
                    "negative_score": (
                        row.negative_score if row.negative_score is not None else 0.0
                    ),
                    "word_count": row.word_count if row.word_count is not None else 0,
                }
            )

    logger.info(
        "GDELT fetch %s %s→%s: %d articles, %.1f MB total",
        ticker,
        date_from,
        date_to,
        len(all_articles),
        total_bytes / 1e6,
    )
    return all_articles, total_bytes


def write_gdelt_news(articles: list[dict], sector: str, engine: Engine) -> int:
    """Filter, rank, and insert GDELT articles into news_raw.

    Applies the relevance filter (source allowlist + wire-service theme gate),
    scores and caps at top 50 per ticker-week, normalises URLs before dedup,
    then inserts new rows.

    Args:
        articles: Output of fetch_gdelt_news.
        sector: ETF ticker for this sector (e.g. 'XLK').
        engine: SQLAlchemy engine.

    Returns:
        Number of new rows inserted.
    """
    if not articles:
        return 0

    candidates = _filter_and_rank(articles)
    if not candidates:
        logger.info("write_gdelt_news sector=%s: 0 candidates after relevance filter", sector)
        return 0

    # Dedup against DB using normalised URLs; store canonical (original) URL
    norm_to_canonical = {_normalize_url(a["url"]): a["url"] for a in candidates}

    with Session(engine) as session:
        # Fetch existing normalised URLs via canonical stored values
        existing_canonical: set[str] = set(
            session.execute(
                select(NewsRaw.url).where(NewsRaw.url.in_(norm_to_canonical.values()))
            ).scalars()
        )
        existing_norms: set[str] = {_normalize_url(u) for u in existing_canonical}

        new_rows = [
            NewsRaw(
                ticker=a["ticker"],
                sector=sector,
                timestamp=a["timestamp"],
                source=a.get("source"),
                title=_format_title(a),
                summary=_format_summary(a),
                url=norm_to_canonical[_normalize_url(a["url"])],
            )
            for a in candidates
            if _normalize_url(a["url"]) not in existing_norms
        ]

        session.add_all(new_rows)
        session.commit()

    logger.info(
        "write_gdelt_news sector=%s: %d new / %d candidates (dupes skipped: %d, "
        "pre-filter total: %d)",
        sector,
        len(new_rows),
        len(candidates),
        len(candidates) - len(new_rows),
        len(articles),
    )
    return len(new_rows)


# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Strip scheme and www. for URL deduplication."""
    norm = re.sub(r"^https?://", "", url or "")
    norm = re.sub(r"^www\.", "", norm)
    return norm


def _source_tier(source: str) -> int:
    """Return source tier (1/2/3) or 0 if not in any allowlist (wire handled separately)."""
    s = (source or "").lower().strip()
    if s in _TIER1:
        return 1
    if s in _TIER2:
        return 2
    if s in _TIER3:
        return 3
    return 0


def _score(tier: int, word_count: int, tone: float) -> float:
    """Combined relevance score. Tier dominates; word count and tone break ties."""
    tier_weight = {1: 3, 2: 2, 3: 1}[tier]
    return tier_weight * 2 + min(word_count / 1000.0, 1.0) + min(abs(tone) / 5.0, 1.0)


def _iso_week(ts: datetime.datetime) -> str:
    """Return 'YYYY-Www' ISO week key for grouping."""
    iso = ts.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _filter_and_rank(articles: list[dict]) -> list[dict]:
    """Apply source allowlist, wire gate, score, and cap at top-N per ticker-week.

    Steps:
      1. Classify each article into Tier 1/2/wire/3/discard.
      2. Wire articles pass only if they carry ECON_STOCKMARKET or ECON_EARNINGSREPORT.
      3. Score survivors: tier × 2 + word_count_norm + tone_norm.
      4. Group by (ticker, ISO week). Within each group keep top _TOP_N_PER_WEEK.
      5. If a week has fewer than _TIER3_FALLBACK_THRESHOLD Tier-1+2 articles,
         admit Tier 3 rows (already scored lower) up to that threshold.
    """
    # Pass 1: classify and score Tier 1+2+wire
    tier12: list[tuple[float, dict]] = []
    tier3_by_week: dict[str, list[tuple[float, dict]]] = defaultdict(list)

    for a in articles:
        src = (a.get("source") or "").lower().strip()
        if src in _WIRE:
            continue  # wire services excluded — see _WIRE comment above
        tier = _source_tier(src)
        if tier == 0:
            continue
        if tier == 3:
            week = _iso_week(a["timestamp"])
            tier3_by_week[week].append((_score(3, a["word_count"], a["tone"]), a))
            continue

        tier12.append((_score(tier, a["word_count"], a["tone"]), a))

    # Pass 2: group Tier 1+2 by week, count, apply top-N cap
    by_week: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for scored_item in tier12:
        week = _iso_week(scored_item[1]["timestamp"])
        by_week[week].append(scored_item)

    result: list[dict] = []
    for week, items in by_week.items():
        items.sort(key=lambda x: -x[0])
        # Tier 3 fallback: top up weeks that are thin on Tier 1+2 coverage
        if len(items) < _TIER3_FALLBACK_THRESHOLD:
            t3 = sorted(tier3_by_week.get(week, []), key=lambda x: -x[0])
            needed = _TIER3_FALLBACK_THRESHOLD - len(items)
            items = items + t3[:needed]
            items.sort(key=lambda x: -x[0])
        for _, a in items[:_TOP_N_PER_WEEK]:
            result.append(a)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _month_chunks(
    date_from: datetime.date, date_to: datetime.date
) -> list[tuple[datetime.date, datetime.date]]:
    """Return (first_day, last_day) pairs for each calendar month in [date_from, date_to]."""
    chunks: list[tuple[datetime.date, datetime.date]] = []
    cursor = date_from.replace(day=1)
    while cursor <= date_to:
        if cursor.month == 12:
            last_of_month = cursor.replace(day=31)
        else:
            last_of_month = cursor.replace(month=cursor.month + 1, day=1) - datetime.timedelta(
                days=1
            )
        chunk_start = max(cursor, date_from)
        chunk_end = min(last_of_month, date_to)
        chunks.append((chunk_start, chunk_end))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return chunks


def _validate_names(names: list[str]) -> None:
    """Raise ValueError if any company name contains characters unsafe for SQL LIKE."""
    for name in names:
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Company name {name!r} contains characters not allowed in SQL LIKE patterns. "
                "Update _SAFE_NAME_RE or fix the name in ticker_company_names.yaml."
            )


def _build_org_conditions(names: list[str]) -> str:
    """Build UPPER() LIKE conditions for all company name variants."""
    conditions: list[str] = []
    for name in names:
        # BigQuery Standard SQL uses \' not '' to escape single quotes in string literals
        safe = name.upper().replace("'", "\\'")
        conditions.append(f"UPPER(Organizations) LIKE '%{safe}%'")
        conditions.append(f"UPPER(V2Organizations) LIKE '%{safe}%'")
    return "(\n    " + "\n    OR ".join(conditions) + "\n  )"


def _build_query(names: list[str], month_start: datetime.date, month_end: datetime.date) -> str:
    """Return a BigQuery SQL query for one calendar month."""
    next_month_start = (
        month_end.replace(day=1).replace(month=month_end.month + 1)
        if month_end.month < 12
        else month_end.replace(year=month_end.year + 1, month=1, day=1)
    )
    org_filter = _build_org_conditions(names)

    return f"""
SELECT
  DATE(_PARTITIONTIME)                                              AS article_date,
  DocumentIdentifier                                                AS url,
  ANY_VALUE(SourceCommonName)                                       AS source,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64))   AS tone,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64))   AS positive_score,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64))   AS negative_score,
  MAX(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(6)] AS INT64))     AS word_count
FROM `{_GDELT_TABLE}`
WHERE _PARTITIONTIME >= TIMESTAMP('{month_start.isoformat()}')
  AND _PARTITIONTIME <  TIMESTAMP('{next_month_start.isoformat()}')
  AND {org_filter}
GROUP BY article_date, url
HAVING MAX(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(6)] AS INT64)) >= {_MIN_WORD_COUNT}
ORDER BY article_date
"""


def _format_title(article: dict) -> str:
    """Synthetic title encoding tone direction and source for the LLM."""
    tone = article.get("tone", 0.0) or 0.0
    direction = "positive" if tone > 0.5 else "negative" if tone < -0.5 else "neutral"
    source = article.get("source") or "unknown"
    ticker = article.get("ticker", "")
    return f"[GDELT/{source}] {ticker}: {direction} coverage (tone {tone:+.1f})"


def _format_summary(article: dict) -> str:
    """Compact tone metrics string for the LLM."""
    tone = article.get("tone", 0.0) or 0.0
    pos = article.get("positive_score", 0.0) or 0.0
    neg = article.get("negative_score", 0.0) or 0.0
    wc = article.get("word_count", 0) or 0
    return f"tone {tone:+.2f} | positive {pos:.1f}% | negative {neg:.1f}% | {wc} words"
