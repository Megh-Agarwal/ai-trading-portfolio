# agentic-portfolio

> A multi-agent LLM signal generation system that produces weekly portfolio rebalances for a universe of 10 US sector ETFs. Three specialized agents (news sentiment, macro regime, prediction-market events) generate views that feed a Black-Litterman optimizer with realistic constraints and transaction cost modeling. Paper-traded live on AWS; backtested rigorously before live deployment.

![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Ingestion layer                                             │
│  yfinance · FRED · Finnhub · Polymarket  ──▶  SQLite cache  │
└────────────────────────┬─────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         Agent layer         │
          │  ┌─────────────────────┐    │
          │  │  News Sentiment     │    │  (Haiku 4.5)
          │  │  Macro Regime       │    │  (Sonnet 4.6)
          │  │  Prediction Markets │    │  (Sonnet 4.6)
          │  └──────────┬──────────┘    │
          └─────────────┼───────────────┘
                        │ views (Q, Ω)
          ┌─────────────▼───────────────┐
          │       Aggregator            │
          │  Black-Litterman views      │
          └─────────────┬───────────────┘
                        │
          ┌─────────────▼───────────────┐
          │       Optimizer             │
          │  BL posterior · CVXPY       │
          │  constraints · risk checks  │
          └─────────────┬───────────────┘
                        │ target weights
          ┌─────────────▼───────────────┐
          │       Execution             │
          │  order gen · fill sim       │
          │  transaction costs          │
          └─────────────────────────────┘
```

All LLM calls are routed through `src/cache.py` (SHA256-keyed) and logged to the `agent_calls` SQLite table.

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd agentic-portfolio
uv sync
cp .env.example .env      # fill in your API keys
uv run python scripts/init_db.py
```

**Required API keys** (set in `.env`):

| Variable | Source |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `FRED_API_KEY` | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `FINNHUB_API_KEY` | [finnhub.io/dashboard](https://finnhub.io/dashboard) |

---

## Project Structure

```
agentic-portfolio/
├── config/          # YAML config files (universe, agents, optimizer, backtest)
├── data/            # SQLite DB and LLM call cache (gitignored)
├── notebooks/       # Per-milestone deliverable notebooks
├── prompts/         # Versioned prompt templates (one per agent)
├── scripts/         # CLI entrypoints (init_db, ingest_*, run_weekly)
├── src/
│   ├── ingestion/   # Data pulls: yfinance, FRED, Finnhub, Polymarket
│   ├── agents/      # LLM agents + base class + Pydantic output schemas
│   ├── aggregator/  # Merge agent views into BL Q and Omega matrices
│   ├── optimizer/   # Equilibrium returns, BL math, CVXPY optimization
│   ├── execution/   # Order generation, fill simulation, state management
│   ├── backtest/    # Walk-forward backtest engine (Milestone 5)
│   ├── eval/        # Metrics, attribution, plotting
│   ├── cache.py     # SHA256-keyed LLM cache — all anthropic calls go here
│   └── exceptions.py
└── tests/
```

---

## Methodology

_Placeholder — will be filled from `decisions.md` as the project matures._

See [`decisions.md`](decisions.md) for all architectural decision records (ADRs), including: universe selection, rebalance cadence, LLM model choices, data sources, and storage rationale.

---

## Results

_Placeholder — backtest and paper-trading results will be added after Milestone 5._

---

## Roadmap

- [x] M0: Project scaffolding
- [ ] M1: Foundation & data layer
- [ ] M2: Agent layer
- [ ] M3: Portfolio optimization & risk
- [ ] M4: Paper trading & execution
- [ ] M5: Backtesting engine
- [ ] M6: AWS deployment
- [ ] M7: Results & write-up

---

## License

MIT
