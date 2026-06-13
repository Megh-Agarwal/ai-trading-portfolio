"""Pydantic output schemas for the three LLM trading agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class NewsSignal(BaseModel):
    """Output schema for the news sentiment agent.

    sector_sentiments maps each ETF ticker (e.g. "XLK") to a sentiment score.
    """

    sector_sentiments: dict[str, float]
    conviction: float = Field(ge=0.0, le=1.0)
    key_themes: list[str]

    @field_validator("sector_sentiments")
    @classmethod
    def sentiments_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for sector, val in v.items():
            if not -1.0 <= val <= 1.0:
                raise ValueError(f"sector {sector!r} sentiment {val} out of [-1, 1]")
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

    implied_prob maps market_id -> current YES probability (0–1).
    sector_impacts maps ETF ticker -> aggregate tilt (-1 to 1) across all markets.
    driving_events maps ETF ticker -> list of market questions that drive the tilt.
    overall_confidence: model's confidence in the sector tilt assignments.
    time_horizon: e.g. "short" / "medium" / "long".
    """

    implied_prob: dict[str, float]
    sector_impacts: dict[str, float]
    driving_events: dict[str, list[str]]
    time_horizon: str
    overall_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("implied_prob")
    @classmethod
    def probs_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for market_id, prob in v.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"market {market_id!r} implied_prob {prob} out of [0, 1]")
        return v

    @field_validator("sector_impacts")
    @classmethod
    def impacts_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for sector, impact in v.items():
            if not -1.0 <= impact <= 1.0:
                raise ValueError(f"sector {sector!r} impact {impact} out of [-1, 1]")
        return v
