from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config import (
    AgentsConfig,
    BacktestConfig,
    OptimizerConfig,
    UniverseConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# Happy-path: all four configs load cleanly
# ---------------------------------------------------------------------------


def test_load_universe():
    cfg = load_config("universe")
    assert isinstance(cfg, UniverseConfig)
    assert cfg.benchmark == "SPY"
    assert len(cfg.tickers) == 10
    assert cfg.ticker_list == [t.ticker for t in cfg.tickers]


def test_load_agents():
    cfg = load_config("agents")
    assert isinstance(cfg, AgentsConfig)
    assert "sentiment" in cfg.agents
    assert "macro" in cfg.agents
    assert "events" in cfg.agents
    assert cfg.agents["sentiment"].model == "claude-haiku-4-5-20251001"
    assert cfg.agents["macro"].model == "claude-sonnet-4-6"


def test_load_optimizer():
    cfg = load_config("optimizer")
    assert isinstance(cfg, OptimizerConfig)
    assert cfg.max_position_weight == 0.25
    assert cfg.vol_target == 0.12
    assert cfg.transaction_cost_bps == 10


def test_optimizer_market_cap_weights():
    cfg = load_config("optimizer")
    assert len(cfg.market_cap_weights) == 10
    assert "XLK" in cfg.market_cap_weights
    assert all(w > 0 for w in cfg.market_cap_weights.values())


def test_optimizer_market_cap_weights_nonpositive_raises(tmp_path, monkeypatch):
    bad = _optimizer_data(market_cap_weights={"XLK": 0.5, "XLF": -0.1})
    _write_and_patch(tmp_path, monkeypatch, "optimizer", bad)
    with pytest.raises(ValidationError, match="market_cap_weights"):
        load_config("optimizer")


def test_load_backtest():
    cfg = load_config("backtest")
    assert isinstance(cfg, BacktestConfig)
    assert cfg.rebalance_frequency == "weekly"
    assert isinstance(cfg.start_date, datetime.date)
    assert cfg.end_date > cfg.start_date


# ---------------------------------------------------------------------------
# Validation: invalid values raise clear errors
# ---------------------------------------------------------------------------


def test_optimizer_max_position_weight_above_one_raises(tmp_path, monkeypatch):
    bad = _optimizer_data(max_position_weight=1.5)
    _write_and_patch(tmp_path, monkeypatch, "optimizer", bad)
    with pytest.raises(ValidationError, match="max_position_weight"):
        load_config("optimizer")


def test_backtest_end_before_start_raises(tmp_path, monkeypatch):
    bad = {"start_date": "2024-06-01", "end_date": "2023-01-01",
           "initial_capital": 1_000_000, "rebalance_frequency": "weekly"}
    _write_and_patch(tmp_path, monkeypatch, "backtest", bad)
    with pytest.raises(ValidationError, match="end_date"):
        load_config("backtest")


def test_agent_negative_temperature_raises(tmp_path, monkeypatch):
    bad = {"agents": {"sentiment": {"model": "claude-haiku-4-5-20251001",
                                    "prompt_template": "prompts/sentiment.txt",
                                    "max_tokens": 512, "temperature": -0.5}}}
    _write_and_patch(tmp_path, monkeypatch, "agents", bad)
    with pytest.raises(ValidationError, match="temperature"):
        load_config("agents")


def test_unknown_config_name_raises():
    with pytest.raises(ValueError, match="Unknown config"):
        load_config("nonexistent")


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("config._CONFIG_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_config("optimizer")


# ---------------------------------------------------------------------------
# Acceptance: changing max_position_weight in YAML is reflected at load time
# ---------------------------------------------------------------------------


def test_max_position_weight_reflects_yaml_value(tmp_path, monkeypatch):
    data = _optimizer_data(max_position_weight=0.15)
    _write_and_patch(tmp_path, monkeypatch, "optimizer", data)
    cfg = load_config("optimizer")
    assert cfg.max_position_weight == 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimum aggregator fields required by OptimizerConfig since Blocker 2 (ADR-012).
# market_cap_weights added for Ticket 3.1 (ADR-016). prior added for Ticket 3.1.
_OPTIMIZER_AGGREGATOR = {
    "aggregator": {
        "max_excess_return_annual": 0.05,
        "omega_base": 0.0001,
        "regime_scale_intercept": 0.75,
        "regime_scale_slope": 0.25,
    },
    "aggregator_weights": {
        "backtest": {"news": 0.57, "macro": 0.43, "polymarket": 0.00},
        "live": {"news": 0.40, "macro": 0.30, "polymarket": 0.30},
    },
    "market_cap_weights": {"XLK": 0.30, "XLF": 0.10, "XLV": 0.10},
    "prior": {"lookback_days": 252},
    "black_litterman": {"tau": 0.05},
    "portfolio": {
        "max_position_weight": 0.25,
        "vol_target": 0.20,
        "turnover_penalty": 0.10,
        "solver_primary": "CLARABEL",
        "solver_fallback": "SCS",
    },
}


def _write_and_patch(tmp_path: Path, monkeypatch, name: str, data: dict) -> None:
    (tmp_path / f"{name}.yaml").write_text(yaml.dump(data))
    monkeypatch.setattr("config._CONFIG_DIR", tmp_path)


def _optimizer_data(**overrides) -> dict:
    """Return a minimal valid optimizer config dict with optional field overrides."""
    base = {
        "tau": 0.05, "risk_aversion": 2.5, "max_position_weight": 0.25,
        "vol_target": 0.12, "turnover_penalty": 0.1, "transaction_cost_bps": 10,
        **_OPTIMIZER_AGGREGATOR,
    }
    base.update(overrides)
    return base
