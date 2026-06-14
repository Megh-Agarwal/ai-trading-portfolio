"""News sentiment agent — Agent 1 of 3."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from agents.base import BaseAgent
from agents.schemas import NewsSignal
from config import load_config
from db.models import NewsRaw, Signal

logger = logging.getLogger(__name__)

_MAX_ARTICLES_PER_SECTOR = 20

_TOOL: dict = {
    "name": "report_sector_sentiment",
    "description": (
        "Report structured sentiment analysis for all 10 sector ETFs based solely "
        "on the provided news articles"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sector_sentiments": {
                "type": "object",
                "description": "Sentiment score per sector ETF",
                "properties": {
                    "XLK": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLF": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLV": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLY": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLP": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLE": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLI": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLB": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLRE": {"type": "number", "minimum": -1, "maximum": 1},
                    "XLU": {"type": "number", "minimum": -1, "maximum": 1},
                },
                "required": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"],
            },
            "sector_conviction": {
                "type": "object",
                "description": (
                    "Per-sector conviction score. Must be 0.2 or below if fewer than "
                    "3 articles exist for that sector."
                ),
                "properties": {
                    "XLK": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLF": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLV": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLY": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLP": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLE": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLI": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLB": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLRE": {"type": "number", "minimum": 0, "maximum": 1},
                    "XLU": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"],
            },
            "key_themes": {
                "type": "array",
                "description": "3 to 5 short phrases describing the dominant market narratives this week",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
            "evidence": {
                "type": "array",
                "description": (
                    "For every sector with |sentiment| > 0.1, cite the specific headline "
                    "that drove the score. If no relevant article exists, sentiment must be 0.0."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "sector": {"type": "string"},
                        "headline": {"type": "string"},
                        "impact": {"type": "string"},
                    },
                    "required": ["sector", "headline", "impact"],
                },
            },
        },
        "required": ["sector_sentiments", "sector_conviction", "key_themes", "evidence"],
    },
}


class NewsAgent(BaseAgent):
    """Reads prior-week news per sector and outputs structured sentiment signals.

    Model: claude-haiku-4-5-20251001 (fast, cheap; adequate for headline classification).
    Output: NewsSignal — per-sector sentiment [-1, 1], per-sector conviction [0, 1].
    Signals table: one row per sector, signal_value = sentiment, confidence = per-sector conviction.
    """

    agent_name = "sentiment"
    _schema_class = NewsSignal
    _tool = _TOOL

    def __init__(self, cache=None) -> None:
        cfg = load_config("agents")
        agent_cfg = cfg.agents["sentiment"]
        prompt_path = Path(__file__).parent.parent.parent / agent_cfg.prompt_template
        super().__init__(
            model_string=agent_cfg.model,
            prompt_template_path=prompt_path,
            cache=cache,
            max_tokens=agent_cfg.max_tokens,
            temperature=agent_cfg.temperature,
        )

    def prepare_input(self, date: datetime.date, db: Engine) -> dict:
        """Query news_raw for the trailing 7 days per sector."""
        week_start = date - datetime.timedelta(days=7)
        start_dt = datetime.datetime.combine(week_start, datetime.time.min)
        end_dt = datetime.datetime.combine(date, datetime.time.max)

        universe = load_config("universe")
        sectors: dict[str, list[dict]] = {}

        with Session(db) as session:
            for ticker_meta in universe.tickers:
                etf = ticker_meta.ticker
                rows = (
                    session.execute(
                        select(NewsRaw)
                        .where(NewsRaw.sector == etf)
                        .where(NewsRaw.timestamp >= start_dt)
                        .where(NewsRaw.timestamp <= end_dt)
                        .order_by(NewsRaw.timestamp.desc())
                        .limit(_MAX_ARTICLES_PER_SECTOR)
                    )
                    .scalars()
                    .all()
                )

                articles = []
                for r in rows:
                    article: dict = {
                        "timestamp": r.timestamp.strftime("%Y-%m-%d") if r.timestamp else None,
                        "ticker": r.ticker,
                        "title": r.title,
                    }
                    if r.summary:
                        article["summary"] = r.summary[:200]
                    articles.append(article)

                sectors[etf] = articles
                logger.debug("Sector %s: %d articles in window", etf, len(articles))

        total = sum(len(v) for v in sectors.values())
        logger.info(
            "prepare_input date=%s  total_articles=%d  sectors=%d",
            date,
            total,
            len(sectors),
        )
        return {
            "analysis_date": date.isoformat(),
            "week_start": week_start.isoformat(),
            "sectors": sectors,
        }

    def _write_signals(
        self,
        date: datetime.date,
        validated: dict,
        call_id: int | None,
        db: Engine,
    ) -> None:
        """Write one Signal row per sector using per-sector conviction; idempotent."""
        sector_conviction = validated["sector_conviction"]
        rows = [
            Signal(
                date=date,
                agent_name=self.agent_name,
                target=sector,
                signal_value=sentiment,
                confidence=sector_conviction.get(sector, 0.0),
                raw_call_id=call_id,
            )
            for sector, sentiment in validated["sector_sentiments"].items()
        ]

        with Session(db) as session:
            session.execute(
                delete(Signal)
                .where(Signal.date == date)
                .where(Signal.agent_name == self.agent_name)
            )
            session.add_all(rows)
            session.commit()

        avg_conv = sum(sector_conviction.values()) / len(sector_conviction) if sector_conviction else 0.0
        logger.info(
            "Wrote %d signal rows  date=%s  agent=%s  avg_conviction=%.2f",
            len(rows),
            date,
            self.agent_name,
            avg_conv,
        )
