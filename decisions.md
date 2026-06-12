# Architectural Decision Records

Each entry: date, context, decision, consequences.

---

## ADR-001 — Universe: 10 sector ETFs, excluding XLC

**Date:** 2026-06-11

**Context:** We needed a tractable, liquid universe for a weekly LLM-driven rebalancer. Options considered: individual S&P 500 stocks (~500 assets), factor ETFs (momentum, value, quality), or SPDR sector ETFs. Individual stocks introduce idiosyncratic risk that is hard to reason about from macro/news signals alone, and covariance estimation at 500 assets is noisy. The 11 SPDR sector ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLRE, XLU, XLC) cover the full market with clean liquidity, but XLC (Communication Services) is dominated by Alphabet and Meta — effectively a large-cap tech overlap with XLK that creates collinearity in the Black-Litterman covariance matrix.

**Decision:** Use the 10 SPDR sector ETFs excluding XLC: XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLRE, XLU.

**Consequences:** Universe is small enough for clean BL math and interpretable agent reasoning. Dropping XLC means Communication Services exposure is absent; acceptable for a research prototype. If XLC correlation with XLK decreases in the future, revisit.

---

## ADR-002 — Weekly rebalance cadence

**Date:** 2026-06-11

**Context:** Rebalance frequency determines transaction costs, signal horizon, and operational complexity. Daily rebalancing amplifies noise in LLM sentiment signals and accumulates transaction costs. Monthly rebalancing misses medium-term macro trends (e.g. a two-week risk-off regime). LLM-derived sentiment signals over news and macro data appear to have a roughly 5–10 day half-life based on related literature.

**Decision:** Rebalance once per week, executed at Friday close (or Monday open if preferred by execution layer).

**Consequences:** ~52 rebalances per year; transaction cost drag is manageable. The pipeline can be scheduled as a single weekly cron job. Intraweek signals are discarded; this is acceptable given the research scope.

---

## ADR-003 — LLM selection: Haiku 4.5 for sentiment, Sonnet 4.6 for macro

**Date:** 2026-06-11

**Context:** Three agents need different capability/cost tradeoffs. The news sentiment agent scores many headlines at high volume — latency and cost per call matter most. The macro regime agent synthesizes multi-source economic data into a structured view — reasoning quality matters more than throughput. The prediction-market agent parses Polymarket event probabilities and maps them to sectors — moderate complexity.

**Decision:** Claude Haiku 4.5 for the news sentiment agent (fast, cheap, sufficient for headline classification). Claude Sonnet 4.6 for the macro regime and prediction-market agents (stronger reasoning, acceptable cost at weekly cadence). All calls routed through `src/cache.py` with SHA256 keying.

**Consequences:** Mixing models complicates the `agent_calls` audit log slightly (model name must be stored per call). Cost for the macro agent is higher but acceptable at 52 calls/year. If Haiku quality proves insufficient for sentiment, upgrade path is a single config change.

---

## ADR-004 — Finnhub for news data; 12-month backtest window

**Date:** 2026-06-11

**Context:** News data options: Bloomberg Terminal (expensive, no API), NewsAPI (no financial focus), Finnhub (financial-specific, free tier covers company/sector news with API access). For the backtest window: too short (3–6 months) misses regime variation; too long (3–5 years) increases look-ahead bias risk and data-pipeline complexity. A 12-month window spans at least one full rate-cycle phase and is practical for a research prototype.

**Decision:** Use Finnhub Python SDK (`finnhub-python`) for news ingestion. Backtest window: 12 months of weekly observations (52 periods).

**Consequences:** Finnhub free tier has rate limits (~60 calls/min); ingestion must respect them. 12-month window means only ~52 training observations for backtest evaluation — enough to demonstrate the methodology, not enough for statistical significance claims. Noted in the write-up.

---

## ADR-005 — SQLite over Postgres for v1

**Date:** 2026-06-11

**Context:** Storage options for price, macro, and agent output data. At weekly cadence the data volume is small: ~10 assets × 52 weeks = ~520 price rows/year; agent outputs are similarly sparse. Options: Postgres (robust, concurrent, but requires a running server), SQLite (file-based, zero-administration), DuckDB (analytical, good for time-series queries). The system runs on a single EC2 instance with no concurrent writers.

**Decision:** SQLite via SQLAlchemy ORM for v1. Migrate to Postgres if concurrent access or replication is ever needed.

**Consequences:** Zero infrastructure overhead; the database is a single file that can be copied for inspection or backup. SQLAlchemy abstracts the engine, so migration to Postgres in a future milestone is a one-line config change. Concurrent write contention is not a concern at this cadence.

---
