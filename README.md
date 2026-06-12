# agentic-portfolio

> Multi-agent LLM signal generation for sector ETF portfolio rebalancing, with Black-Litterman optimization and live paper trading.

**Status:** 🚧 Work in progress (Milestone 1 of 7)

## What this is

A weekly-rebalanced portfolio of 10 US sector ETFs driven by three specialized LLM agents (news sentiment, macro regime, prediction-market events) feeding a Black-Litterman optimizer with realistic constraints and transaction costs. Backtested over the past 12 months and paper-traded live on AWS.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd agentic-portfolio
uv sync
cp .env.example .env  # fill in API keys
uv run python scripts/init_db.py
```

Required API keys: `ANTHROPIC_API_KEY`, `FRED_API_KEY`, `FINNHUB_API_KEY`.

## Architecture

See `CLAUDE.md` for the system architecture and `decisions.md` for the rationale behind major design choices.

## Roadmap

- [x] M0: Project scaffolding
- [ ] M1: Foundation & data layer
- [ ] M2: Agent layer
- [ ] M3: Portfolio optimization & risk
- [ ] M4: Paper trading & execution
- [ ] M5: Backtesting engine
- [ ] M6: AWS deployment
- [ ] M7: Results & write-up

## License

MIT