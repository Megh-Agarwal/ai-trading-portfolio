# agentic-portfolio

> A multi-agent LLM signal generation system that produces weekly portfolio rebalances for a universe of 10 US sector ETFs. Three specialized agents (news sentiment, macro regime, prediction-market events) generate views that feed a Black-Litterman optimizer with realistic constraints and transaction cost modeling. Paper-traded live on AWS; backtested rigorously before live deployment.

![Status](https://img.shields.io/badge/status-live-brightgreen)

**This system runs live and unattended every week on AWS.** → [View the live dashboard](https://megh-agarwal-portfolio-dashboard.hf.space)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Ingestion layer                                             │
│  yfinance · FRED · Alpha Vantage · Finnhub · Polymarket  ──▶  SQLite cache  │
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
| `GCP_PROJECT_ID` + `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account with BigQuery read access — used for GDELT news backfill |

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
│   ├── ingestion/   # Data pulls: yfinance, FRED, Alpha Vantage, Finnhub, Polymarket
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

53-week out-of-sample backtest (2025-06-13 → 2026-06-12), four portfolios run in parallel:

| Portfolio | Total Return | Notes |
|---|---|---|
| LLM (Full) | +21.93% | All three agents active |
| No-LLM | +23.78% | BL prior only, no agent views |
| Equal Weight | — | 1/N baseline |
| SPY | — | Passive benchmark |

The −1.85% LLM alpha gap is attributable to signal quality (97%) rather than transaction costs (3%). Macro regime modulation is the primary underperformer — the next research iteration addresses this. See `decisions.md` for the full analysis (ADR-022, ADR-023).

Live paper trading started 2026-06-28. Results update automatically every Sunday.

---

## Roadmap

- [x] M1: Foundation & data layer
- [x] M2: Agent layer
- [x] M3: Portfolio optimization & risk
- [x] M4: Paper trading & execution
- [x] M5: Backtesting engine (102 weeks × 4 portfolios, $4.61 total LLM cost)
- [x] M6: AWS deployment (EC2 + cron + API + public dashboard)
- [ ] M7: Signal quality improvements & write-up

---

## License

MIT
