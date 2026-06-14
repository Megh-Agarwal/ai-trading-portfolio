"""Pydantic output schemas for the three LLM trading agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NewsSignal(BaseModel):
    """Output schema for the news sentiment agent.

    sector_sentiments maps each ETF ticker to a sentiment score.
    sector_conviction maps each ETF ticker to a per-sector conviction score.
    evidence cites specific headlines for any |sentiment| > 0.1.
    """

    sector_sentiments: dict[str, float]
    sector_conviction: dict[str, float]
    key_themes: list[str]
    evidence: list[dict]

    @field_validator("sector_sentiments")
    @classmethod
    def sentiments_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for sector, val in v.items():
            if not -1.0 <= val <= 1.0:
                raise ValueError(f"sector {sector!r} sentiment {val} out of [-1, 1]")
        return v

    @field_validator("sector_conviction")
    @classmethod
    def convictions_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for sector, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"sector {sector!r} conviction {val} out of [0, 1]")
        return v


class MacroRegimeSignal(BaseModel):
    """Output schema for the macro regime agent.

    reasoning: chain-of-thought scratchpad (stored in agent_calls.response_json).
    rationale: 2-3 sentence clean summary for logging and display.
    """

    reasoning: str
    regime: Literal["risk_on", "risk_off", "neutral"]
    rate_outlook: Literal["rising", "falling", "stable"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class PolymarketSignal(BaseModel):
    """Output schema for the prediction-market events agent.

    implied_probs maps market_id -> current YES probability (0–1).
    sector_tilts maps ETF ticker -> aggregate tilt (-1 to 1) across all markets.
    driving_events: list of {sector, market_question, reasoning} for |tilt| >= 0.05.
    overall_confidence: model's confidence in the sector tilt assignments.
    time_horizon: "short" / "medium" / "long".
    """

    judgments: str
    implied_probs: dict[str, float]
    sector_tilts: dict[str, float]
    driving_events: list[dict]
    time_horizon: Literal["short", "medium", "long"]
    overall_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("implied_probs")
    @classmethod
    def probs_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for market_id, prob in v.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"market {market_id!r} implied_probs {prob} out of [0, 1]")
        return v

    @field_validator("sector_tilts")
    @classmethod
    def tilts_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for sector, tilt in v.items():
            if not -1.0 <= tilt <= 1.0:
                raise ValueError(f"sector {sector!r} tilt {tilt} out of [-1, 1]")
        return v
