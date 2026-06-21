# Architectural Decision Records

Each entry: date, context, decision, consequences.

---

## ADR-016 — Market-cap weights for Black-Litterman equilibrium prior

**Date:** 2026-06-15

**Context:** `compute_equilibrium_returns` (Ticket 3.1) requires market-portfolio weights w_mkt to compute the BL implied equilibrium return π = λΣw_mkt. The BL model uses the global market portfolio; for a 10-sector ETF universe the best available proxy is the S&P 500 sector composition as published by SSGA (State Street Global Advisors) in their SPY factsheet.

**Decision:** Store SPY sector weights in `config/optimizer.yaml` under `market_cap_weights` (ticker → raw percentage). `compute_equilibrium_returns` renormalises the weights to sum=1 at call time, so raw SSGA percentages can be passed directly without manually adjusting for the excluded XLC sector. Refresh quarterly when SPY rebalances; changes are a one-line YAML edit with no code changes required.

**Consequences:** The prior is mildly stale between quarterly refreshes. Sector weights drift slowly (±1–2% per quarter) so the impact on equilibrium returns is small. Ledoit-Wolf shrinkage (not sample covariance) is used by default for the covariance estimate because T/N ≈ 25 (252 days / 10 sectors) is enough for LW to be reliable but too small for the sample matrix to be well-conditioned without shrinkage.

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

## ADR-010 — Alpha Vantage as primary historical news source alongside Finnhub

**Date:** 2026-06-13

**Context:** The Finnhub free tier ignores the `_from` date parameter on the company_news endpoint and returns only the last ~5 days of articles regardless of the requested range. This was discovered when `validate_historical_depth` consistently returned 0 months even for AAPL. The 12-18 month backtest window requires genuinely historical news. Alpha Vantage's NEWS_SENTIMENT endpoint (free tier: 25 calls/day) honors date range parameters and returns articles going back ~2 years.

**Decision:** Add `src/ingestion/alpha_vantage_news.py` as a parallel adapter. Keep `src/ingestion/news.py` (Finnhub) unchanged — it correctly captures recent articles and may be useful for forward paper-trading updates. The AV adapter batches all 10 constituent tickers per ETF into one API call (10 calls total for the full universe), staying well within the 25 calls/day free tier limit. Fan-out logic in `fetch_av_news` creates one NewsRaw row per (article, matched-ticker) pair, so an article covering both AAPL and MSFT produces two rows. `write_av_news` deduplicates on (url, ticker) pairs rather than url alone, because the same article legitimately maps to multiple tickers.

**Consequences:** The `news_raw` table now receives data from two sources (source column distinguishes them). Historical backfill uses AV; ongoing collection can use either. Free tier caps at 200 articles per call — sectors with high news volume may see truncation over an 18-month window; a warning is logged when this occurs. Upgrading to AV premium raises the limit to 1000 per call. `ALPHA_VANTAGE_API_KEY` added to `.env.example`.

---

## ADR-011 — Views aggregation: signal-to-return mapping and agent weights

**Date:** 2026-06-13

**Context:** `build_views` (Ticket 2.5) converts three agent outputs on a [-1, 1] scale into Black-Litterman views (Q vector, Omega diagonal matrix). Two design choices required explicit justification because they are non-identifiable from data without a calibration exercise: (1) the scale factor mapping a ±1 signal to an expected excess return, and (2) the relative weights assigned to the three agents.

**Decision:**

1. **Signal-to-return mapping:** A ±1 signal represents a maximum conviction directional view and maps to ±5 % annualised expected excess return, de-annualised to weekly (÷52). Rationale: 5 % annualised excess return is roughly one standard deviation of annual sector-ETF active returns, making a unit signal mean "one-sigma conviction". The parameter is named `_MAX_EXCESS_RETURN_ANNUAL` and lives in `src/aggregator/views.py` so it can be adjusted for calibration without touching logic.

2. **Agent weights (40/30/30):** News sentiment is given slightly more weight (0.40) because it is the highest-frequency, most directly price-relevant signal. Macro regime (0.30) and Polymarket events (0.30) carry equal weight because both are structural signals operating on a slower timescale. This split is defensible but not estimated — it should be treated as a prior and calibrated during backtesting (M5).

3. **Macro as confidence multiplier:** The macro regime does not provide per-sector directional views. Instead, it scales the magnitude of all views via `regime_scale = 0.75 + 0.25 × regime_float`, giving [0.50, 0.75, 1.00] for [risk_off, neutral, risk_on]. This means risk-off regimes shrink both Q (smaller bets) and increase Omega (more uncertainty), producing smaller position changes in the BL optimizer — capturing the intuition that macro headwinds should dampen conviction in any single-stock or sector call.

4. **Omega construction:** `Omega_ii = OMEGA_BASE / max(conviction_i, MIN_CONVICTION)` where `OMEGA_BASE = 0.0001` (~1 bp² weekly at unit conviction). The inverse-proportional form is the simplest link between conviction and view uncertainty consistent with BL theory. A more sophisticated calibration would set `OMEGA = tau × P × Sigma × P'` but that requires the covariance matrix from the optimizer — this simplified form decouples the aggregator from the optimizer layer.

**Consequences:** All four parameters (`_MAX_EXCESS_RETURN_ANNUAL`, default weights, `_OMEGA_BASE`, `_MIN_CONVICTION`) are module-level constants in `views.py` that can be swept during M5 backtest calibration. The signal-to-return mapping is the most sensitive: changing it from 5 % to 10 % doubles all position tilts. This must be documented prominently in the eventual paper's methodology section.

---

## ADR-012 — Separate backtest and live weights for Polymarket agent

**Date:** 2026-06-14

**Context:** The `polymarket_raw` table only contains data from June 2026 (the date the ingestion script was first run). All historical backtest dates prior to that have `current_prob = null` for every curated market, meaning Agent 3 (Polymarket) always outputs zero tilts during backtesting. A 30% weight on a perpetually-zero agent degrades backtest quality without adding information.

**Decision:** Add separate `backtest` and `live` weight sets to `config/optimizer.yaml` under `aggregator_weights`. The `build_views` function takes a `mode` parameter ("live" or "backtest") and selects the corresponding weight set. Backtest weights: news=0.57, macro=0.43, polymarket=0.00. Live weights unchanged: news=0.40, macro=0.30, polymarket=0.30. The backtest weights redistribute the 30% polymarket allocation proportionally between news and macro.

**Consequences:** Backtest results reflect a two-agent system (news + macro). Live forward trading uses all three agents equally with Polymarket as the third source. Any comparison of backtest vs. live Sharpe ratios must account for this structural difference — it is not a model improvement but a data availability constraint. Documented in methodology section as a known limitation.

---

## ADR-013 — MacroAgent news digest restricted to XLF and XLI

**Date:** 2026-06-14

**Context:** The macro agent's prompt includes a news digest to provide narrative context for the quantitative indicators. Including all 10 sectors would roughly triple the news token count per macro call (~1,500 additional tokens at Sonnet 4.6 pricing), adding ~$0.02/call for marginal benefit. The macro agent's primary job is regime classification from FRED quantitative data; news is supplementary context.

**Decision:** Restrict macro news digest to XLF (Financials) and XLI (Industrials). Rationale: financials are the most macro-sensitive sector (directly affected by rate decisions, credit conditions, and liquidity); industrials are the best leading indicator of the real economy (PMI, durable goods, capex). Both sectors have high information density per article relative to macro regime classification.

**Consequences:** A major tech regulation event (e.g. XLK) or energy shock (XLE) may not surface in macro reasoning unless it also appears in XLF or XLI news. This is an accepted limitation — such events typically propagate into quantitative indicators (VIX spike, rate moves) faster than the weekly news digest would surface them. If macro miss-classification due to absent sector news becomes a recurring issue, extend the digest to XLK and XLE.

---

## ADR-014 — rate_outlook stored but not used in v1 portfolio construction

**Date:** 2026-06-14

**Context:** `MacroAgent._write_signals` writes two rows per date: `target="macro_regime"` (used by aggregator) and `target="rate_outlook"` (not used). The rate outlook carries information relevant to rate-sensitive sectors (XLF benefits from rising rates; XLRE and XLU are hurt). Wiring it into the aggregator requires sector-specific rate sensitivity weights — a calibration decision that belongs in M5 once we have backtest data.

**Decision:** Store `rate_outlook` in the signals table but do not consume it in `build_views` for v1. The data is preserved for two purposes: (1) M5 attribution analysis — test whether weeks where macro agent predicted rising rates correlate with XLF outperformance vs. XLRE underperformance; (2) v2 aggregator enhancement — wire rate_outlook into per-sector Q adjustments once sensitivity weights are calibrated from backtest.

**Consequences:** rate_outlook is effectively dead signal in v1 but costs nothing to store. If attribution analysis (M5) confirms meaningful predictive power, the upgrade path is to add a `rate_sensitivity` vector per sector to the aggregator config and include it in Q calculation. This is one config stanza and a few lines of code in `build_views`.

---

## ADR-015 — Per-sector conviction replaces single conviction scalar

**Date:** 2026-06-14

**Context:** The original `NewsSignal` schema had a single `conviction: float` applied uniformly to all 10 sectors. This was architecturally incorrect: an agent that reads 15 XLK articles and 1 XLU article should report high conviction for XLK and low conviction for XLU, not a blended scalar that applies to both. The uniform conviction also prevented the aggregator from correctly weighting sectors with thin news coverage.

**Decision:** Update `NewsSignal` to output `sector_conviction: dict[str, float]` (one value per ETF ticker) instead of `conviction: float`. Each value is independently constrained to [0, 1]. The tool schema enforces `sector_conviction ≤ 0.2` for sectors with fewer than 3 articles by prompt instruction (not schema enforcement, since the API cannot count articles). The signals table stores per-sector conviction in `Signal.confidence` from this session forward, giving the aggregator correct per-sector uncertainty inputs.

**Consequences:** Any historical signal rows written before this change carry the old single-conviction value replicated across all sectors — they are still valid for aggregation but lose the per-sector granularity. For the backtest window, all signals will be regenerated via cached API calls so historical rows will reflect per-sector conviction. The `evidence` field added to `NewsSignal` enforces grounding: any sector with |sentiment| > 0.1 must cite a specific headline, preventing the model from outputting non-zero scores based on market memory.

---

## ADR-018 — Risk circuit breakers: pre-trade and post-trade checks

**Date:** 2026-06-15

**Context:** A paper trading system has no real P&L exposure, so automated risk controls might seem unnecessary. However, the discipline of building explicit checks before and after every optimization is what separates a research project from a toy system. Each check encodes a specific failure mode and is a concrete talking point in any interview or write-up — "the system had a 15% rolling drawdown circuit breaker that halted rebalancing for one week" signals risk management maturity in a way that hand-waving cannot.

**Decision:** Implement four checks in `src/optimizer/risk_checks.py` as individual functions each returning a `RiskCheckResult` dataclass. Checks are orchestrated by `run_all_risk_checks` which runs them all, logs every result (triggered or not) to `risk_events`, then applies triggered actions in sequence. Drawdown is a circuit breaker that overrides all other actions.

Four checks:
1. **max_position** (pre-trade): clip any weight exceeding `portfolio.max_position_weight` and renormalize. Catches residual violations from the solver's clip+renorm that floating-point rounding might leave behind.
2. **max_turnover** (pre-trade): if L1 turnover exceeds `risk.max_single_rebalance_turnover` (default 0.50), blend new and previous weights 50/50. Prevents a single large signal shift from forcing a massive trade in one week.
3. **drawdown_circuit_breaker** (post-trade state): if rolling 20-day drawdown of `portfolio_snapshot.total_value` falls below −15%, return previous weights unchanged — skip the rebalance entirely. Enforces "don't fight a falling knife" discipline.
4. **realized_vol** (post-trade state): if annualised realized vol over the lookback window exceeds `vol_target × vol_breach_multiplier` (default 1.5×), blend weights 20% toward equal-weight — partial deleveraging without abandoning positions.

All thresholds live in `config/optimizer.yaml` under `risk:`. No constants are hardcoded.

**Consequences:** The drawdown circuit breaker will occasionally halt rebalancing during legitimate recoveries (false positives). This is the preferred direction of error — missing a rebalance costs less than doubling down into a crash. The 15% threshold is conservative and should be disclosed in the methodology as a design choice, not a calibrated parameter. All risk events are written to the `risk_events` table for post-hoc audit and attribution analysis.

---

## ADR-019 — SPY sector weights corrected against SSGA published data (2026-06-21)

**Date:** 2026-06-21

**Context:** The `market_cap_weights` values in `config/optimizer.yaml` were set during Ticket 3.1 without verification against the SSGA SPY factsheet (flagged as a to-do). A cross-check against SSGA's current published sector allocation revealed one material discrepancy: XLV (Health Care) was stored as 12.2% but SSGA reports 9.47%, a gap of −2.73 percentage points. Because these weights are renormalized after excluding XLC, the XLV overstatement was also suppressing XLK's effective equilibrium weight by ~2.4% in the normalized prior. The remaining nine sectors were all within ±0.7% of SSGA actuals.

**Decision:** Replace all ten values with the current SSGA figures (as published June 2026). The change reflects Health Care's genuine decline in S&P 500 composition over the past year (GLP-1 enthusiasm faded, pharma repricing headwinds). All values updated to four decimal precision.

| Ticker | Old | New | Delta |
|--------|-----|-----|-------|
| XLK | 0.323 | 0.3291 | +0.61pp |
| XLF | 0.131 | 0.1259 | −0.51pp |
| XLV | 0.122 | 0.0947 | **−2.73pp** |
| XLY | 0.102 | 0.0986 | −0.34pp |
| XLP | 0.059 | 0.0525 | −0.65pp |
| XLE | 0.040 | 0.0402 | +0.02pp |
| XLI | 0.089 | 0.0902 | +0.12pp |
| XLB | 0.026 | 0.0209 | −0.51pp |
| XLRE | 0.025 | 0.0195 | −0.55pp |
| XLU | 0.026 | 0.0254 | −0.06pp |

**Consequences:** The BL equilibrium prior will now correctly underweight Health Care and more accurately reflect Technology's ~37% share of the ex-XLC S&P 500. Effect on portfolio output is modest for most sectors but XLV's equilibrium return π was materially overstated before this fix. Next refresh due Q3 2026 (September) when SPY rebalances.

---

## ADR-017 — Transaction cost assumption: 3 bps one-way (conservative)

**Date:** 2026-06-15

**Context:** The backtester and optimizer turnover penalty both need a one-way transaction cost estimate for sector ETF trades. Actual bid-ask spreads for liquid SPDR sector ETFs (XLK, XLF, etc.) are typically sub-1bp in normal market conditions. However, a model that overstates strategy performance by assuming unrealistically low costs is more dangerous than one that understates it.

**Decision:** Use 1bp spread + 2bp slippage = 3bps one-way total. This is deliberately conservative relative to observed reality. The 1bp spread reflects a wider-than-typical half-spread; the 2bp slippage accounts for market impact on a small institutional account ($1–5M AUM). Both parameters live in `config/optimizer.yaml` under `transaction_costs` and are never hardcoded. Positions where |Δw| ≤ 0.001 (0.1% weight change) are treated as rounding noise and incur no cost — this avoids charging for micro-adjustments from solver tolerance.

**Consequences:** Backtest Sharpe ratios and net returns will be slightly pessimistic relative to what a real implementation would likely achieve. This is the preferred direction of error for a research prototype. If live paper-trading data shows actual costs consistently below 3bps, the YAML can be updated without code changes. The conservative assumption should be disclosed in the methodology section of any write-up.
