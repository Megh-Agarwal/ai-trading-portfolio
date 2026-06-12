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

## ADR-006 — Forward-fill for macro frequency alignment

**Date:** 2026-06-11

**Context:** Price data is daily, but the macro series have mixed frequencies: T10Y2Y, DGS10, VIXCLS, and DTWEXBGS are daily (business days only); CPIAUCSL and UNRATE are monthly; ICSA is weekly (Thursday release). The macro regime agent needs a consistent daily view so it can be called on any given trading day without special-casing frequency.

**Decision:** Reindex each series to a daily date range and forward-fill (no backfill). Fetch with a 90-day buffer before the requested start date so that monthly series always have a seed value on day one of the window. Rows with unavoidable leading NaN (series starts after the window) are dropped with a warning.

**Consequences:** Each daily row carries the most recently *released* value — this correctly reflects the information available on that date and avoids look-ahead bias. The 1–4 week publication lag in CPI and UNRATE is preserved implicitly. Downstream agents should not treat daily macro values as "same-day" updates — the lag is part of the signal.

---

## ADR-009 — Polymarket integration: Gamma API, sector mappings, and historical-data limitation

**Date:** 2026-06-12

**Context:** Prediction-market signals are the third input to the Black-Litterman aggregator. Three design choices were required: (1) which Polymarket API to use for data ingestion; (2) how to curate a stable, macro-relevant set of markets from a platform where markets expire on resolution; (3) how to handle the absence of a simple bulk-history download for the 12-month backtest.

**Decisions:**

1. **Gamma API over py-clob-client for data fetching.** The Polymarket CLOB client (`py-clob-client`) is optimised for order execution and requires API-key authentication even for reads. The Gamma REST API (`gamma-api.polymarket.com`) is public and unauthenticated; it exposes market metadata, current prices (`outcomePrices`), and volume in a single GET. `requests` is used directly — no SDK wrapper needed for read-only ingestion.

2. **Manual curation into `config/polymarket_markets.yaml`.** The unique IP of the project is the sector-impact mapping: each market carries explicit `positive_if_yes` / `negative_if_yes` / `mixed` signals per ETF, derived from economic theory (rate-sensitivity, commodity exposure, consumer cyclicality). Markets were selected using three filters: (a) >$100k cumulative volume for liquidity; (b) clear 1–12 month horizon so signals are actionable on a weekly cadence; (c) each of the 10 ETF sectors covered by at least one market. 13 markets curated at launch; YAML must be refreshed quarterly as markets resolve.

3. **Current-state only for live ingestion; CLOB prices-history for backtest.** Polymarket does not offer a bulk CSV history download. `fetch_market_prices(condition_id, start, end)` queries the CLOB prices-history endpoint (`clob.polymarket.com/prices-history`) which returns daily YES-token prices going back to market creation — usable for backtest but on a per-market basis. For markets that were live during the backtest window, this yields a full probability time series; for markets that postdate the window, we fall back to the current snapshot as a static prior (acknowledged limitation in the write-up).

**Consequences:** The Gamma API does not require credentials (no `.env` entry needed for Polymarket). Market IDs in the YAML will expire as markets resolve; `ingest_polymarket.py` logs a warning for each skipped market. The backtest agent receives either a dynamic probability series (preferred) or a static prior (fallback) per market — the aggregator must handle both cases. This is documented as a known limitation in the methodology section.

---

## ADR-008 — Finnhub news ingestion: rate limiting, dedup strategy, and monthly chunking

**Date:** 2026-06-11

**Context:** Finnhub's free tier permits ~60 API calls/minute and returns up to ~12 months of company news per ticker. Three design choices needed to be locked down: (1) how to enforce the rate limit without a queue or thread pool; (2) how to deduplicate articles across re-runs without adding a unique constraint to the DB; (3) how to structure the date window across 100 tickers × 12 months without hitting pagination limits.

**Decision:**
1. **Rate limiting:** Module-level `_last_call_ts` global + `_rate_limit()` guard before every API call. Sleep only as long as needed. No queue, no threading — ingestion is single-threaded by design.
2. **Dedup by URL:** Query existing URLs in batch before insert (`NewsRaw.url.in_(urls)`), then only insert rows not already present. No `UNIQUE` constraint added to `news_raw.url` because SQLite `ON CONFLICT IGNORE` would silently swallow errors for other columns — application-layer dedup is explicit and testable.
3. **Monthly chunking:** Split the 12-month backfill window into calendar-month slices. This keeps each API call well within Finnhub's undocumented per-request result cap and makes resumable partial ingestion straightforward.

**Consequences:** Wall time for a full backfill (1200 calls) is ~22 minutes. Re-runs are safe (idempotent). Historical depth is validated before the full loop fires — if Finnhub returns < 10 months the window is shortened and a warning is logged. Application-layer dedup adds one SELECT per batch of articles but avoids schema coupling.

---

## ADR-007 — ETF holdings source and refresh cadence

**Date:** 2026-06-11

**Context:** Agent 1 (news sentiment) needs a list of individual stock tickers to pull news for, aggregated to the sector level. Options: (a) scrape SSGA's ETF pages directly — reliable but brittle HTML parsing; (b) use yfinance `Ticker.funds_data.top_holdings` — simpler, depends on Yahoo Finance's data feed; (c) hardcode a static list — zero network dependency but goes stale. The top 10 holdings of each SPDR sector ETF together represent the bulk of each sector's weight. Holdings change slowly — rebalances happen quarterly.

**Decision:** Use `yfinance.Ticker(etf).funds_data.top_holdings` to fetch the top 10 holdings per sector ETF and cache the result in `config/sector_holdings.yaml`. Refresh by re-running `scripts/update_holdings.py` each quarter. At the time of writing, the 100 tickers across 10 ETFs have zero cross-sector overlap.

**Consequences:** The YAML is the source of truth for Agent 1 — if Yahoo Finance's funds data is stale or unavailable, the cached YAML still works. The validation step in `update_holdings.py` logs any cross-sector overlap so analysts can decide whether to deduplicate. If yfinance changes its `funds_data` API, the scraper is a one-file change.

---
