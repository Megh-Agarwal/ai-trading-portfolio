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

# Cap articles per sector to control token usage; Haiku context window is large
# but we want fast, focused calls. 20 headlines cover the most material news.
_MAX_ARTICLES_PER_SECTOR = 20

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "news_sentiment.txt"


class NewsAgent(BaseAgent):
    """Reads prior-week news per sector and outputs structured sentiment signals.

    Model: claude-haiku-4-5-20251001 (fast, cheap; adequate for headline classification).
    Output: NewsSignal — per-sector sentiment [-1, 1], conviction [0, 1], key_themes.
    Signals table: one row per sector, signal_value = sentiment, confidence = conviction.
    """

    agent_name = "sentiment"
    _schema_class = NewsSignal

    def __init__(self, cache=None) -> None:
        cfg = load_config("agents")
        agent_cfg = cfg.agents["sentiment"]
        super().__init__(
            model_string=agent_cfg.model,
            prompt_template_path=_PROMPT_PATH,
            cache=cache,
            max_tokens=agent_cfg.max_tokens,
            temperature=agent_cfg.temperature,
        )

    def prepare_input(self, date: datetime.date, db: Engine) -> dict:
        """Query news_raw for the trailing 7 days per sector.

        Args:
            date: Rebalance date (exclusive upper bound — we read news UP TO this date).
            db: SQLAlchemy engine.

        Returns:
            Dict with analysis_date, week_start, and sectors mapping each ETF
            ticker to a list of article dicts.
        """
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
        """Write one Signal row per sector; delete any prior rows for this date first."""
        conviction = validated["conviction"]
        rows = [
            Signal(
                date=date,
                agent_name=self.agent_name,
                target=sector,
                signal_value=sentiment,
                confidence=conviction,
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

        logger.info(
            "Wrote %d signal rows  date=%s  agent=%s  conviction=%.2f",
            len(rows),
            date,
            self.agent_name,
            conviction,
        )
