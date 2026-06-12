# Agentic Portfolio Rebalancing System

## Project Overview

A multi-agent LLM signal generation system that produces weekly portfolio rebalances for a universe of 10 US sector ETFs. Three specialized agents (news sentiment, macro regime, prediction-market events) generate views that feed a Black-Litterman optimizer with realistic constraints and transaction cost modeling. Paper-traded live on AWS once v1 ships; backtested rigorously before live deployment.

This is a personal research project and resume artifact. Code quality matters; over-engineering does not. Prefer flat, readable modules over deep abstractions.

## Tech Stack

- Python 3.11+
- Dependency management: `uv` (not poetry, not pip)
- Storage: SQLite via SQLAlchemy ORM
- Optimization: CVXPY
- LLM API: Anthropic SDK (Claude Haiku 4.5 + Sonnet 4.6)
- Data: yfinance, fredapi, finnhub-python, py-clob-client
- Testing: pytest

## Architecture

Data flow (left to right):

ingestion → cache → agents → aggregator → optimizer → execution → snapshot
↓          ↓        ↓          ↓            ↓           ↓
SQLite ←──────┴────────┴──────────┴────────────┴───────────┘

- `src/ingestion/`: Pulls raw data from yfinance, FRED, Finnhub, Polymarket. Writes to SQLite.
- `src/cache.py`: SHA256-keyed file cache for ALL LLM calls. Logs metadata to `agent_calls` table.
- `src/agents/`: Three LLM agents + base class + JSON output schemas (pydantic).
- `src/aggregator/`: Deterministic merger of agent outputs into Black-Litterman views (Q, Omega).
- `src/optimizer/`: Equilibrium returns, BL math, CVXPY constrained optimization, risk checks.
- `src/execution/`: Order generation, fill simulation, transaction costs, state management.
- `src/backtest/`: Walk-forward backtest engine (added in Milestone 5).
- `src/eval/`: Metrics, attribution, plotting.
- `scripts/`: Entrypoints (init_db, ingest_*, run_weekly, run_backtest).
- `config/`: YAML files for universe, agents, optimizer, backtest.
- `prompts/`: Versioned prompt templates (one per agent).
- `notebooks/`: Per-milestone deliverable notebooks (01_data_exploration.ipynb, etc.).
- `tests/`: pytest unit and smoke tests.

## Critical Rules

These are non-negotiable. Violating them will cost real money or break reproducibility.

1. **ALL LLM API calls go through `src/cache.py`**. Never instantiate `anthropic.Anthropic()` outside that module. Direct SDK calls bypass the cache and bypass the `agent_calls` audit log.
2. **All database access via SQLAlchemy session**, not raw SQL strings. Migrations are the only exception.
3. **All tunable parameters live in `config/*.yaml`**, never hardcoded in source. Model strings, constraint values, agent weights, etc.
4. **Every meaningful tradeoff gets a 2-3 line entry in `decisions.md`** with date, context, decision, consequences. This is the source for the eventual write-up's methodology section.
5. **Idempotency:** any script that writes to SQLite must be safely re-runnable. Use upsert semantics, not insert-only.
6. **Type hints required** on all public functions. Use `from __future__ import annotations` at the top of modules.
7. **No look-ahead bias** anywhere. The backtest engine (M5) will enforce this systematically, but ingestion code should already respect it — never join "future" data into a past-dated row.

## Coding Conventions

- Line length: 100 chars (configured in pyproject.toml)
- Linter/formatter: `ruff` for both
- Docstrings: Google style on public functions
- Logging: `logging` module, not `print()`. Module-level logger: `logger = logging.getLogger(__name__)`.
- Imports: grouped (stdlib, third-party, local), alphabetized within group
- Errors: raise specific exceptions, not bare `Exception`. Custom exceptions in `src/exceptions.py`.
- Tests: pytest, one test file per source module, `tests/test_<module>.py`

## Workflow

- Work is organized as GitHub Issues, grouped by Milestone (M1-M7).
- One ticket = one branch = one PR = one commit (or squash-merge to one).
- Branch naming: `<ticket-number>-<short-slug>`, e.g., `1.4-yfinance-ingestion`.
- Commit messages: conventional commits format. Reference issue number. Example: `feat(ingestion): add yfinance price loader (#4)`
- Before opening a PR, run: `uv run ruff check src/ tests/` and `uv run pytest`. Both must pass.
- Acceptance criteria in the ticket are gates, not suggestions. Verify each box before closing.

## Common Commands

```bash
uv sync                                    # install dependencies
uv run pytest                              # run tests
uv run ruff check src/ tests/              # lint
uv run ruff format src/ tests/             # format
uv run python scripts/init_db.py           # initialize SQLite schema
uv run python scripts/ingest_prices.py     # populate price data
uv run python scripts/run_weekly.py        # full weekly pipeline (M4+)
```

## What This Project Is Not (Anti-Goals)

- **Not** a high-frequency or intraday trading system. Weekly rebalance only.
- **Not** real-money trading. Paper trading only, even in production.
- **Not** an over-engineered enterprise system. Single SQLite file, single EC2 instance, single repo. No microservices, no message queues, no Kubernetes.
- **Not** a backtest-only project. Live forward paper trading on AWS is part of the deliverable.
- **Not** a learning-the-LLM project. Use frontier models via API with good prompts; no fine-tuning, no embeddings databases unless explicitly required.

## Where to Look for "Why" Questions

- Architectural decisions: `decisions.md`
- Tunable parameters: `config/*.yaml`
- Prompt design: `prompts/*.txt`
- Anything else: ask the user before assuming.