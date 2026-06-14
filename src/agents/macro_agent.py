"""Macro regime agent — Agent 2 of 3."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from agents.base import BaseAgent, _REGIME_TO_FLOAT, _RATE_OUTLOOK_TO_FLOAT
from agents.schemas import MacroRegimeSignal
from config import load_config
from db.models import Macro, NewsRaw, Signal

logger = logging.getLogger(__name__)

# How far back to query macro data. 90 days for rolling averages; 400 for CPI YoY.
_MACRO_LOOKBACK_DAYS = 400
_NEWS_LOOKBACK_DAYS = 7
_MAX_NEWS_ARTICLES = 10  # per sector in macro news digest

_TOOL: dict = {
    "name": "report_macro_regime",
    "description": "Classify the current macroeconomic regime based on provided indicators",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step analysis of the macro indicators, 150-300 words. "
                    "Write this first before setting labels."
                ),
            },
            "regime": {
                "type": "string",
                "enum": ["risk_on", "risk_off", "neutral"],
                "description": "Current market regime",
            },
            "rate_outlook": {
                "type": "string",
                "enum": ["rising", "falling", "stable"],
                "description": "Near-term rate direction",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "0.8+ only when all indicators agree. 0.3-0.5 for mixed signals.",
            },
            "rationale": {
                "type": "string",
                "description": "2-3 sentence plain-English summary for a portfolio manager",
            },
        },
        "required": ["reasoning", "regime", "rate_outlook", "confidence", "rationale"],
    },
}


class MacroAgent(BaseAgent):
    """Classifies the macro regime and rate outlook from FRED series + macro news.

    Model: claude-sonnet-4-6 (more reasoning capacity for multi-indicator synthesis).
    Output: MacroRegimeSignal — regime, rate_outlook, confidence, reasoning, rationale.
    Signals table:
        target="macro_regime"  signal_value in {-1.0, 0.0, 1.0}
        target="rate_outlook"  signal_value in {-1.0, 0.0, 1.0}
    """

    agent_name = "macro"
    _schema_class = MacroRegimeSignal
    _tool = _TOOL

    def __init__(self, cache=None) -> None:
        cfg = load_config("agents")
        agent_cfg = cfg.agents["macro"]
        prompt_path = Path(__file__).parent.parent.parent / agent_cfg.prompt_template
        super().__init__(
            model_string=agent_cfg.model,
            prompt_template_path=prompt_path,
            cache=cache,
            max_tokens=agent_cfg.max_tokens,
            temperature=agent_cfg.temperature,
        )

    def prepare_input(self, date: datetime.date, db: Engine) -> dict:
        """Build structured macro input for the LLM.

        Queries:
        - Last 400 days of all 7 FRED series (for derived features including CPI YoY)
        - Last 7 days of XLF + XLI news (macro news digest)

        Raw 30-day time series are excluded — only pre-computed derived features
        are passed to the LLM to reduce token usage (~1,500 tokens saved per call).

        Args:
            date: Rebalance date (inclusive upper bound).
            db: SQLAlchemy engine.

        Returns:
            Dict with analysis_date, derived_features, and macro_news_digest.
        """
        start_400d = date - datetime.timedelta(days=_MACRO_LOOKBACK_DAYS)
        start_90d = date - datetime.timedelta(days=90)
        start_30d = date - datetime.timedelta(days=30)
        news_start = date - datetime.timedelta(days=_NEWS_LOOKBACK_DAYS)

        with Session(db) as session:
            macro_rows = (
                session.execute(
                    select(Macro)
                    .where(Macro.date >= start_400d)
                    .where(Macro.date <= date)
                    .order_by(Macro.date)
                )
                .scalars()
                .all()
            )

            news_rows = (
                session.execute(
                    select(NewsRaw)
                    .where(NewsRaw.sector.in_(["XLF", "XLI"]))
                    .where(NewsRaw.timestamp >= datetime.datetime.combine(news_start, datetime.time.min))
                    .where(NewsRaw.timestamp <= datetime.datetime.combine(date, datetime.time.max))
                    .order_by(NewsRaw.timestamp.desc())
                    .limit(_MAX_NEWS_ARTICLES * 2)
                )
                .scalars()
                .all()
            )

        derived = _compute_derived_features(macro_rows, date, start_90d, start_30d)
        news_digest = _format_news_digest(news_rows, _MAX_NEWS_ARTICLES)

        total_articles = sum(len(v) for v in news_digest.values())
        logger.info(
            "prepare_input date=%s  macro_rows=%d  news_articles=%d",
            date,
            len(macro_rows),
            total_articles,
        )

        return {
            "analysis_date": date.isoformat(),
            "derived_features": derived,
            "macro_news_digest": news_digest,
        }

    def _write_signals(
        self,
        date: datetime.date,
        validated: dict,
        call_id: int | None,
        db: Engine,
    ) -> None:
        """Write two Signal rows (macro_regime + rate_outlook); idempotent."""
        confidence = validated["confidence"]
        rows = [
            Signal(
                date=date,
                agent_name=self.agent_name,
                target="macro_regime",
                signal_value=_REGIME_TO_FLOAT[validated["regime"]],
                confidence=confidence,
                raw_call_id=call_id,
            ),
            Signal(
                date=date,
                agent_name=self.agent_name,
                target="rate_outlook",
                signal_value=_RATE_OUTLOOK_TO_FLOAT[validated["rate_outlook"]],
                confidence=confidence,
                raw_call_id=call_id,
            ),
        ]

        with Session(db) as session:
            session.execute(
                delete(Signal)
                .where(Signal.date == date)
                .where(Signal.agent_name == self.agent_name)
            )
            session.add_all(rows)
            session.commit()

        logger.info(
            "Wrote 2 signal rows  date=%s  regime=%s  rate_outlook=%s  confidence=%.2f",
            date,
            validated["regime"],
            validated["rate_outlook"],
            confidence,
        )


# ---------------------------------------------------------------------------
# Private helpers — data shaping
# ---------------------------------------------------------------------------


def _compute_derived_features(
    rows: list,
    date: datetime.date,
    start_90d: datetime.date,
    start_30d: datetime.date,
) -> dict:
    """Compute rolling statistics from raw macro rows."""
    if not rows:
        return {}

    df = pd.DataFrame(
        [(r.date, r.series_id, r.value) for r in rows],
        columns=["date", "series_id", "value"],
    )
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    pivot = pivot.sort_index()

    features: dict = {}
    cutoff_90d = pd.Timestamp(start_90d)
    cutoff_30d = pd.Timestamp(start_30d)
    cutoff_365d = pd.Timestamp(date - datetime.timedelta(days=365))

    def _latest(series_id: str) -> float | None:
        if series_id not in pivot.columns:
            return None
        col = pivot[series_id].dropna()
        return float(col.iloc[-1]) if len(col) > 0 else None

    def _avg(series_id: str, since: pd.Timestamp) -> float | None:
        if series_id not in pivot.columns:
            return None
        col = pivot[series_id].dropna()
        window = col[col.index >= since]
        return float(window.mean()) if len(window) > 0 else None

    def _change_since(series_id: str, since: pd.Timestamp) -> float | None:
        if series_id not in pivot.columns:
            return None
        col = pivot[series_id].dropna()
        window = col[col.index >= since]
        if len(window) < 2:
            return None
        return float(window.iloc[-1] - window.iloc[0])

    # VIX
    vix_current = _latest("VIXCLS")
    if vix_current is not None:
        features["vix_current"] = round(vix_current, 2)
    vix_30d_chg = _change_since("VIXCLS", cutoff_30d)
    if vix_30d_chg is not None:
        features["vix_30d_change"] = round(vix_30d_chg, 2)
    vix_90d_avg = _avg("VIXCLS", cutoff_90d)
    if vix_90d_avg is not None:
        features["vix_90d_avg"] = round(vix_90d_avg, 2)

    # T10Y2Y
    t10y2y_current = _latest("T10Y2Y")
    if t10y2y_current is not None:
        features["t10y2y_current"] = round(t10y2y_current, 3)
    t10y2y_90d_avg = _avg("T10Y2Y", cutoff_90d)
    if t10y2y_90d_avg is not None:
        features["t10y2y_90d_avg"] = round(t10y2y_90d_avg, 3)
    if t10y2y_current is not None and t10y2y_90d_avg is not None:
        features["t10y2y_vs_90d_avg"] = round(t10y2y_current - t10y2y_90d_avg, 3)

    # DGS10
    dgs10_current = _latest("DGS10")
    if dgs10_current is not None:
        features["dgs10_current"] = round(dgs10_current, 3)
    dgs10_30d_chg = _change_since("DGS10", cutoff_30d)
    if dgs10_30d_chg is not None:
        features["dgs10_30d_change"] = round(dgs10_30d_chg, 3)

    # CPI YoY
    if "CPIAUCSL" in pivot.columns:
        cpi = pivot["CPIAUCSL"].dropna()
        if len(cpi) > 0:
            features["cpi_current"] = round(float(cpi.iloc[-1]), 3)
            cpi_prior = cpi[cpi.index <= cutoff_365d]
            if len(cpi_prior) > 0:
                prior_val = float(cpi_prior.iloc[-1])
                if prior_val != 0:
                    features["cpi_yoy_pct"] = round(
                        (float(cpi.iloc[-1]) - prior_val) / prior_val * 100, 2
                    )

    # UNRATE
    unrate_current = _latest("UNRATE")
    if unrate_current is not None:
        features["unrate_current"] = round(unrate_current, 1)
    unrate_3m_chg = _change_since("UNRATE", cutoff_90d)
    if unrate_3m_chg is not None:
        features["unrate_3m_change"] = round(unrate_3m_chg, 1)

    # ICSA
    icsa_current = _latest("ICSA")
    if icsa_current is not None:
        features["icsa_current"] = int(icsa_current)
    icsa_4w_avg = _avg("ICSA", pd.Timestamp(date - datetime.timedelta(days=28)))
    if icsa_4w_avg is not None:
        features["icsa_4w_avg"] = int(icsa_4w_avg)

    # USD index
    usd_current = _latest("DTWEXBGS")
    if usd_current is not None:
        features["usd_index_current"] = round(usd_current, 2)
    usd_30d_chg = _change_since("DTWEXBGS", cutoff_30d)
    if usd_30d_chg is not None:
        features["usd_index_30d_change"] = round(usd_30d_chg, 2)

    return features


def _format_news_digest(rows: list, max_per_sector: int) -> dict[str, list[dict]]:
    """Return macro news digest grouped by sector, capped per sector."""
    digest: dict[str, list[dict]] = {"XLF": [], "XLI": []}
    counts: dict[str, int] = {"XLF": 0, "XLI": 0}

    for r in rows:
        sector = r.sector or ""
        if sector not in digest:
            continue
        if counts[sector] >= max_per_sector:
            continue
        article: dict = {
            "date": r.timestamp.strftime("%Y-%m-%d") if r.timestamp else None,
            "ticker": r.ticker,
            "title": r.title,
        }
        if r.summary:
            article["summary"] = r.summary[:150]
        digest[sector].append(article)
        counts[sector] += 1

    return digest
