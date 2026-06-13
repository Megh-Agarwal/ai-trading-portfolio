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
    """Output schema for the macro regime agent."""

    regime: Literal["risk_on", "risk_off", "neutral"]
    rate_outlook: Literal["rising", "falling", "stable"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class PolymarketSignal(BaseModel):
    """Output schema for the prediction-market events agent.

    implied_prob maps market_id -> YES probability (0–1).
    sector_impacts maps ETF ticker -> net directional impact (-1 to 1).
    """

    implied_prob: dict[str, float]
    sector_impacts: dict[str, float]
    time_horizon: str

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
