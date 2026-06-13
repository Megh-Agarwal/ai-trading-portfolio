"""Polymarket events agent — Agent 3 of 3."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from agents.base import BaseAgent
from agents.schemas import PolymarketSignal
from config import load_config
from db.models import PolymarketRaw, Signal
from ingestion.polymarket import load_curated_markets

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "polymarket_events.txt"
_LOOKBACK_DAYS = 30


class PolymarketAgent(BaseAgent):
    """Translates Polymarket event probabilities into per-sector equity tilts.

    Model: claude-haiku-4-5-20251001 (task is partly mechanical rule application).
    Output: PolymarketSignal — per-sector tilt [-1,1], driving_events, time_horizon.
    Signals table: one row per sector, signal_value = sector tilt, confidence = overall_confidence.
    """

    agent_name = "events"
    _schema_class = PolymarketSignal

    def __init__(self, cache=None) -> None:
        cfg = load_config("agents")
        agent_cfg = cfg.agents["events"]
        super().__init__(
            model_string=agent_cfg.model,
            prompt_template_path=_PROMPT_PATH,
            cache=cache,
            max_tokens=agent_cfg.max_tokens,
            temperature=agent_cfg.temperature,
        )
        self._curated = load_curated_markets()

    def prepare_input(self, date: datetime.date, db: Engine) -> dict:
        """Build structured Polymarket input for the LLM.

        For each curated market:
        - Latest implied_prob from polymarket_raw up to `date`
        - Earliest implied_prob in the 30-day window (as "30d ago" trend proxy)
        - Volume, end_date, days_to_resolution from the latest snapshot
        - Sector impacts and confidence_rating from the YAML config

        Markets with no DB row are included with current_prob=null so the LLM
        knows to treat them as data-absent.

        Args:
            date: Rebalance date (inclusive upper bound for DB queries).
            db: SQLAlchemy engine.

        Returns:
            Dict with analysis_date and markets list.
        """
        date_dt = datetime.datetime.combine(date, datetime.time.max)
        start_30d_dt = datetime.datetime.combine(date - datetime.timedelta(days=_LOOKBACK_DAYS), datetime.time.min)

        curated_by_id = {m["market_id"]: m for m in self._curated}
        market_ids = list(curated_by_id.keys())

        with Session(db) as session:
            # Latest snapshot per market (max timestamp ≤ date)
            subq = (
                select(
                    PolymarketRaw.market_id,
                    func.max(PolymarketRaw.timestamp).label("max_ts"),
                )
                .where(PolymarketRaw.market_id.in_(market_ids))
                .where(PolymarketRaw.timestamp <= date_dt)
                .group_by(PolymarketRaw.market_id)
                .subquery()
            )
            latest_rows = (
                session.execute(
                    select(PolymarketRaw).join(
                        subq,
                        (PolymarketRaw.market_id == subq.c.market_id)
                        & (PolymarketRaw.timestamp == subq.c.max_ts),
                    )
                )
                .scalars()
                .all()
            )

            # Earliest snapshot in 30d window per market (trend proxy)
            subq_30d = (
                select(
                    PolymarketRaw.market_id,
                    func.min(PolymarketRaw.timestamp).label("min_ts"),
                )
                .where(PolymarketRaw.market_id.in_(market_ids))
                .where(PolymarketRaw.timestamp >= start_30d_dt)
                .where(PolymarketRaw.timestamp <= date_dt)
                .group_by(PolymarketRaw.market_id)
                .subquery()
            )
            earliest_30d_rows = (
                session.execute(
                    select(PolymarketRaw).join(
                        subq_30d,
                        (PolymarketRaw.market_id == subq_30d.c.market_id)
                        & (PolymarketRaw.timestamp == subq_30d.c.min_ts),
                    )
                )
                .scalars()
                .all()
            )

        latest_by_id = {r.market_id: r for r in latest_rows}
        early_by_id = {r.market_id: r for r in earliest_30d_rows}

        markets_input = []
        for mid, cfg_market in curated_by_id.items():
            latest = latest_by_id.get(mid)
            early = early_by_id.get(mid)

            current_prob = round(float(latest.implied_prob), 4) if latest else None
            prob_30d_ago = round(float(early.implied_prob), 4) if early else None
            volume_usd = float(latest.volume) if latest and latest.volume else 0.0
            end_date = latest.end_date if latest else None
            days_to_resolution = (end_date - date).days if end_date else None

            entry: dict = {
                "market_id": mid,
                "question": cfg_market.get("question", ""),
                "category": cfg_market.get("category", ""),
                "confidence_rating": cfg_market.get("confidence", "medium"),
                "current_prob": current_prob,
                "prob_30d_ago": prob_30d_ago,
                "volume_usd": round(volume_usd, 0),
                "days_to_resolution": days_to_resolution,
                "sector_impacts": cfg_market.get("sector_impacts", {}),
            }
            markets_input.append(entry)

        has_data = sum(1 for m in markets_input if m["current_prob"] is not None)
        logger.info(
            "prepare_input date=%s  markets_with_data=%d/%d",
            date,
            has_data,
            len(markets_input),
        )

        return {
            "analysis_date": date.isoformat(),
            "markets": markets_input,
        }

    def _write_signals(
        self,
        date: datetime.date,
        validated: dict,
        call_id: int | None,
        db: Engine,
    ) -> None:
        """Write one Signal row per sector; idempotent."""
        confidence = validated["overall_confidence"]
        rows = [
            Signal(
                date=date,
                agent_name=self.agent_name,
                target=sector,
                signal_value=tilt,
                confidence=confidence,
                raw_call_id=call_id,
            )
            for sector, tilt in validated["sector_impacts"].items()
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
            "Wrote %d signal rows  date=%s  agent=%s  confidence=%.2f",
            len(rows),
            date,
            self.agent_name,
            confidence,
        )
