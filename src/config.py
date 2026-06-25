from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Literal, overload

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TickerMeta(BaseModel):
    ticker: str
    sector: str
    name: str | None = None


class UniverseConfig(BaseModel):
    benchmark: str
    tickers: list[TickerMeta]

    @property
    def ticker_list(self) -> list[str]:
        return [t.ticker for t in self.tickers]


class AgentConfig(BaseModel):
    model: str
    prompt_template: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)


class AgentsConfig(BaseModel):
    agents: dict[str, AgentConfig]


class AggregatorParams(BaseModel):
    max_excess_return_annual: float = Field(gt=0)
    omega_base: float = Field(gt=0)
    regime_scale_intercept: float
    regime_scale_slope: float


class AgentWeights(BaseModel):
    news: float = Field(ge=0, le=1)
    macro: float = Field(ge=0, le=1)
    polymarket: float = Field(ge=0, le=1)


class AggregatorWeights(BaseModel):
    backtest: AgentWeights
    live: AgentWeights


class PriorConfig(BaseModel):
    lookback_days: int = Field(gt=0)


class BlackLittermanConfig(BaseModel):
    tau: float = Field(gt=0)


class TransactionCostsConfig(BaseModel):
    spread_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    min_trade_threshold: float = Field(ge=0)


class PortfolioConfig(BaseModel):
    max_position_weight: float = Field(gt=0, le=1)
    vol_target: float = Field(gt=0, le=1)
    turnover_penalty: float = Field(ge=0)
    solver_primary: str
    solver_fallback: str


class RiskConfig(BaseModel):
    max_single_rebalance_turnover: float = Field(gt=0)
    max_drawdown_threshold: float = Field(gt=0)
    drawdown_lookback_days: int = Field(gt=0)
    vol_breach_multiplier: float = Field(gt=0)
    vol_deleveraging_blend: float = Field(gt=0, le=1)


class OptimizerConfig(BaseModel):
    tau: float = Field(gt=0)
    risk_aversion: float = Field(gt=0)
    max_position_weight: float = Field(gt=0, le=1)
    vol_target: float = Field(gt=0, le=1)
    turnover_penalty: float = Field(ge=0)
    transaction_cost_bps: float = Field(ge=0)
    aggregator: AggregatorParams
    aggregator_weights: AggregatorWeights
    market_cap_weights: dict[str, float]
    prior: PriorConfig
    black_litterman: BlackLittermanConfig
    transaction_costs: TransactionCostsConfig
    portfolio: PortfolioConfig
    risk: RiskConfig

    @model_validator(mode="after")
    def check_market_cap_weights(self) -> OptimizerConfig:
        for ticker, w in self.market_cap_weights.items():
            if w <= 0:
                raise ValueError(f"market_cap_weights[{ticker!r}] = {w} must be positive")
        return self


class BacktestConfig(BaseModel):
    start_date: datetime.date
    end_date: datetime.date
    initial_capital: float = Field(gt=0)
    rebalance_frequency: Literal["daily", "weekly", "monthly"]

    @model_validator(mode="after")
    def end_after_start(self) -> BacktestConfig:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseModel]] = {
    "universe": UniverseConfig,
    "agents": AgentsConfig,
    "optimizer": OptimizerConfig,
    "backtest": BacktestConfig,
}


@overload
def load_config(name: Literal["universe"]) -> UniverseConfig: ...
@overload
def load_config(name: Literal["agents"]) -> AgentsConfig: ...
@overload
def load_config(name: Literal["optimizer"]) -> OptimizerConfig: ...
@overload
def load_config(name: Literal["backtest"]) -> BacktestConfig: ...


def load_config(name: str) -> BaseModel:
    """Load and validate a config file from the config/ directory.

    Args:
        name: Config name without extension — one of universe, agents, optimizer, backtest.

    Returns:
        Validated Pydantic model for the requested config.

    Raises:
        ValueError: Unknown config name.
        FileNotFoundError: YAML file not found.
        pydantic.ValidationError: Config fails schema validation.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown config {name!r}. Valid names: {sorted(_REGISTRY)}")
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    model = _REGISTRY[name].model_validate(raw)
    logger.debug("Loaded config %r from %s", name, path)
    return model
