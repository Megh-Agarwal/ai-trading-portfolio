# Architecture Explained — Agentic Portfolio Rebalancing System

A beginner-friendly walkthrough for developers who know how to code but are new to quantitative finance and LLM engineering patterns.

---

## Table of Contents

1. [The One-Paragraph Version](#1-the-one-paragraph-version)
2. [The Cast of Characters (Glossary)](#2-the-cast-of-characters-glossary)
3. [File-by-File Walkthrough](#3-file-by-file-walkthrough)
4. [What Each Agent Actually Does (Deep Dive)](#4-what-each-agent-actually-does-deep-dive)
5. [How the Three Agents' Opinions Get Combined](#5-how-the-three-agents-opinions-get-combined)
6. [How "Black-Litterman" Actually Works, No Formulas](#6-how-black-litterman-actually-works-no-formulas)
7. [How Target Weights Get Decided (the Optimizer)](#7-how-target-weights-get-decided-the-optimizer)
8. [How a Trade Actually Happens (Execution)](#8-how-a-trade-actually-happens-execution)
9. [A Single Real Walkthrough, Start to Finish](#9-a-single-real-walkthrough-start-to-finish)
10. [Every Configuration Parameter, What It Controls, and Its Current Value](#10-every-configuration-parameter-what-it-controls-and-its-current-value)
11. [Known Limitations and Open Questions](#11-known-limitations-and-open-questions)

---

## 1. The One-Paragraph Version

Once a week, this system reads the latest financial news, economic data, and prediction-market odds, then asks three separate AI models to each form an opinion about which sectors of the stock market look attractive. Those three opinions get merged into a single blended forecast that is then used to nudge a mathematically-derived "what would the whole market hold?" baseline — the result is a set of target percentages for how to split a $1,000,000 paper-trading portfolio across 10 exchange-traded funds (ETFs) that each represent one slice of the US economy (tech, energy, healthcare, etc.). A constrained optimizer decides the exact percentages while making sure no single ETF gets more than 25% and the overall risk stays within a preset ceiling. The system then calculates how many shares to buy or sell of each ETF to reach those percentages, simulates the trade fills with realistic transaction costs, and records everything to a SQLite database. The whole pipeline costs roughly $0.06 in LLM API fees on the first run; subsequent runs for the same date hit a local file cache and cost $0.

---

## 2. The Cast of Characters (Glossary)

**Sector ETF** — An exchange-traded fund that holds shares in many companies within one industry segment. Buying one share of XLK, for example, gives you exposure to Apple, Microsoft, Nvidia, and 70+ other technology companies simultaneously. The 10 ETFs in this system are: XLK (Tech), XLF (Financials), XLV (Health Care), XLY (Consumer Discretionary), XLP (Consumer Staples), XLE (Energy), XLI (Industrials), XLB (Materials), XLRE (Real Estate), XLU (Utilities).

**Prediction market** — A marketplace where people stake real money on yes/no questions about future events. Example question: "Will the Federal Reserve cut interest rates before July 2026?" People who think YES will happen buy YES shares; people who think NO will happen buy NO shares. The more people buy YES, the higher the YES share price climbs. When the event resolves, YES shares pay out $1 and NO shares pay $0 (or vice versa). Because real money is at stake, prediction markets tend to be well-calibrated — the crowd has a financial incentive to be right, not just confident.

**Implied probability** — The probability embedded in a prediction market's current price. If YES shares currently cost $0.80, that means: to break even, you need at least an 80% chance of YES happening. So the market is collectively "implying" there's an 80% chance the event occurs. An implied probability of 0.03 (3%) means the market thinks it's very unlikely. An implied probability of 0.97 (97%) means the market thinks it's nearly certain. Implied probability is just the YES price expressed as a fraction — no formula needed.

**Sentiment** — How bullish or bearish a piece of news or an agent's view is toward a given sector. Represented as a number from −1.0 (very bearish — expecting price to fall) to +1.0 (very bullish — expecting price to rise). Zero means no strong view.

**Conviction** — How confident an agent is in its sentiment. A number from 0.0 to 1.0. Low conviction (e.g. 0.2) means "I have a weak opinion"; high conviction (e.g. 0.8) means "I feel quite sure." Conviction controls how much the agent's opinion is allowed to move the final portfolio allocation — see [Section 5](#5-how-the-three-agents-opinions-get-combined).

**Regime** — The overall "mood" of the economy at a given moment. The macro agent classifies the current regime as one of three states: `risk_on` (investors are feeling confident, buying growth stocks), `risk_off` (investors are fearful, fleeing to safe assets), or `neutral` (mixed signals, no strong tilt). The regime acts as a multiplier on the other agents' signals — see [Section 5](#5-how-the-three-agents-opinions-get-combined).

**Equilibrium return** — The answer to the question "what annual return would each ETF need to offer so that a rational market participant would hold the market's current weights?" If the market has 33% in tech (as SPY does), equilibrium logic implies the market collectively expects tech to compensate for its risk. Mathematically it is `λ × Σ × w_mkt` (risk aversion × covariance × market weights). This is the "prior" in Black-Litterman. You do not need to understand the formula — see [Section 6](#6-how-black-litterman-actually-works-no-formulas).

**Prior** — The starting belief before any agent says anything. In this system, the prior is the equilibrium return described above: "if I had no information at all, I'd hold the market portfolio." The word "prior" comes from Bayesian statistics, where you start with a belief and update it with evidence.

**Posterior** — The updated belief after blending the prior with the agents' opinions. The prior says "hold the market"; the agents say "tech looks better than energy this week"; the posterior is somewhere in between, weighted by how confident the agents are. See [Section 6](#6-how-black-litterman-actually-works-no-formulas).

**Black-Litterman** — A mathematical framework (invented at Goldman Sachs in 1990) for blending a market-implied starting portfolio with analyst opinions in a statistically principled way. The key insight is that it lets you say "I think tech will outperform by 3% annually, and I'm 60% confident" rather than just picking weights arbitrarily. See [Section 6](#6-how-black-litterman-actually-works-no-formulas) for a plain-English explanation.

**Covariance** — A measure of how two assets move together. If tech and energy tend to go up and down at the same time, they have positive covariance. If they tend to move in opposite directions, negative covariance. A **covariance matrix** (called Σ in the code) captures all pairwise relationships among the 10 ETFs simultaneously — it's a 10×10 grid of numbers. The system needs it to understand total portfolio risk, because owning two correlated assets is riskier than owning two uncorrelated ones.

**Volatility** — How much an asset's price fluctuates. A sector with 20% annual volatility might move ±20% in a typical year. The system imposes a 12% annualised portfolio volatility ceiling — if all agents pile into a single risky sector, the optimizer is allowed to ignore the resulting weight until the risk is within bounds.

**Basis points (bps)** — A unit used to talk about small percentages in finance. 1 basis point = 0.01%. 10 basis points = 0.10%. Used here for transaction costs: the system models a 1 bp spread + 2 bp slippage = 3 bp total one-way cost per trade (i.e., 0.03% of the dollar amount traded).

**Turnover** — How much of the portfolio changes in a single rebalance. Measured as the sum of all absolute weight changes. Turnover of 0.30 means 30% of the portfolio moved (15% sold, 15% bought). High turnover is expensive because every trade has a cost. The optimizer has a built-in penalty that discourages unnecessary trading.

**Slippage** — The difference between the price you expect to pay and the price you actually pay due to market impact. If you want to buy $10,000 of XLK and moving that amount slightly moves the price against you, you pay a little more. This is modeled here as a fixed 2 bps estimate.

**Long-only** — The portfolio only holds positive positions (no short selling — betting on prices going down). Every ETF weight must be ≥ 0.

**Rebalancing** — Adjusting the portfolio to match target percentages. If XLK grows from 20% to 27% of the portfolio because tech prices rose, a rebalance would sell enough XLK to bring it back to the target.

**Backtest vs live mode** — Backtest mode simulates what the system would have done on historical dates (no real money, no real trades, uses historical data). Live mode runs on today's date and would eventually paper-trade on a real broker. The two modes weight the Polymarket agent differently because prediction-market historical data isn't available for past dates.

---

## 3. File-by-File Walkthrough

### `src/ingestion/`

**`prices.py`** — Pulls daily price data for all 10 ETFs plus SPY (the benchmark) from Yahoo Finance via `yfinance`. Stores one row per (date, ticker) in the `prices` table with columns for open, high, low, close, adjusted close, and volume. Uses `auto_adjust=False` and reshapes Yahoo's multi-level column format into a flat long table. Without this file, there would be no price history for computing returns or executing orders.

**`macro.py`** — Fetches 7 economic time series from the Federal Reserve's FRED database: VIX (fear index), T10Y2Y (yield curve shape), DGS10 (10-year interest rate), CPIAUCSL (inflation index), UNRATE (unemployment rate), ICSA (weekly jobless claims), and DTWEXBGS (dollar index). Monthly and weekly series get forward-filled to daily frequency. Requires a `FRED_API_KEY` environment variable. Without this, the macro agent has no data to classify the economic regime.

**`news.py`** — Fetches company-level news headlines from Finnhub's API. Designed for weekly refreshes (last 7 days only, because Finnhub's free tier doesn't go back further regardless of date parameters). Deduplicates by URL at the application level. Without this, the news sentiment agent has nothing to read.

**`alpha_vantage_news.py`** — An alternative news ingester that fetches from Alpha Vantage's `NEWS_SENTIMENT` API. Used specifically for the 18-month historical backfill because Finnhub can't go back that far. Critical quirks: (1) the free tier only allows one ticker per API call — multi-ticker requests silently return empty; (2) commas in URLs must be literal, not URL-encoded. Runs one ticker at a time, pausing 12 seconds between calls to respect rate limits.

**`holdings.py`** — Fetches the top-10 stock holdings for each sector ETF from Yahoo Finance. Maps ETF → list of constituent stocks (e.g., XLK → ["AAPL", "MSFT", "NVDA", ...]). Used by `news.py` to know which company news belongs to which sector. Refreshed quarterly since ETF holdings change slowly.

**`polymarket.py`** — Fetches current betting-market data from Polymarket's public Gamma API (no authentication required). For each of 13 curated questions (like "Will the Fed cut rates by July?"), it records the current probability that "YES" resolves, trading volume, and resolution date. A critical detail: the `outcomePrices` field comes back as a JSON-encoded *string* (e.g. `'["0.97","0.03"]'`), not an array — the code calls `json.loads()` to parse it. Without this, the Polymarket agent has no event probabilities to work with.

---

### `src/agents/`

**`base.py`** — The abstract base class all three agents inherit from. Handles the pattern that every agent shares: call `prepare_input()` to build a data payload, use the SHA256 cache to avoid redundant LLM calls, call the Anthropic API with a forced tool call (so the model *must* return structured JSON), check if the response was truncated, validate the response against a Pydantic schema, then write signals to the database. Any agent that inherits from this gets all of that for free.

**`schemas.py`** — Pydantic models defining the exact JSON structure each agent must return. Acts as a contract between the LLM and the rest of the system. The three schemas are `NewsSignal` (per-sector sentiments + per-sector conviction + key themes + evidence), `MacroRegimeSignal` (regime label + rate outlook + confidence + reasoning), and `PolymarketSignal` (per-market probabilities + sector tilts + driving events + confidence). If the LLM returns something that doesn't match these schemas, validation fails and the agent gracefully produces a zero stub instead.

**`news_agent.py`** — The news sentiment agent. See [Section 4](#4-what-each-agent-actually-does-deep-dive) for a full walkthrough.

**`macro_agent.py`** — The macro regime agent. See [Section 4](#4-what-each-agent-actually-does-deep-dive).

**`polymarket_agent.py`** — The prediction-market events agent. See [Section 4](#4-what-each-agent-actually-does-deep-dive).

**`pipeline.py`** — Runs all three agents in sequence for a given date, then calls the aggregator. If one agent fails, it logs the error, inserts neutral zero signals for that agent, and continues. Only if all three fail does it raise an error. Returns a summary dict with per-agent status, latency, cost, and the combined views.

---

### `src/aggregator/`

**`views.py`** — Takes all the signals from the `signals` table and combines them into two outputs: `Q` (a vector of 10 numbers, one per sector, representing how much the agents collectively think each sector will outperform or underperform) and `Omega` (a 10×10 diagonal matrix representing how uncertain those views are). This is the aggregator. See [Section 5](#5-how-the-three-agents-opinions-get-combined) for a full explanation of what happens inside.

---

### `src/optimizer/`

**`equilibrium.py`** — Computes the "prior": what would a rational market participant hold if they had no opinions? Uses one year of historical prices to estimate the 10×10 covariance matrix (how the ETFs move relative to each other), then combines it with SPY's sector weights to back out implied equilibrium returns. Uses Ledoit-Wolf shrinkage (a more stable version of standard covariance estimation when you have many assets relative to the number of observations).

**`black_litterman.py`** — The math engine that blends the prior (equilibrium) with the views (agents). Takes in π (equilibrium returns), Σ (covariance), Q (agent views), Ω (view uncertainty), and τ (a scaling factor), and outputs μ* (posterior expected returns) and Σ* (posterior covariance). See [Section 6](#6-how-black-litterman-actually-works-no-formulas) for the plain-English explanation.

**`portfolio.py`** — The constrained optimizer. Given the posterior expected returns and the posterior covariance, finds the set of ETF weights that maximises expected return minus risk minus trading cost penalty, subject to: weights sum to 1, no negative weights, no weight exceeds 25%, and portfolio volatility ≤ 12%. Uses the CVXPY library with CLARABEL as primary solver (and SCS as fallback). If the volatility constraint makes the problem unsolvable, drops that constraint and re-solves on the remaining constraints ("infeasible_relaxed" path). See [Section 7](#7-how-target-weights-get-decided-the-optimizer).

**`risk_checks.py`** — Four safety checks run before and after every rebalance:
- `drawdown_circuit_breaker`: if the portfolio has lost ≥15% from its recent peak, stop rebalancing entirely (freeze the portfolio until conditions improve).
- `realized_vol`: if the portfolio's actual day-to-day volatility over the past 20 days exceeds 1.5× the 12% target (i.e., ≥18%), blend 20% toward equal weights to reduce risk.
- `max_position`: if any weight exceeds 25% (e.g., due to small rounding), clip it and renormalize.
- `max_turnover`: if a single rebalance would change more than 50% of the portfolio, blend 50/50 with the previous weights instead.

**`pipeline.py`** (`optimizer/pipeline.py`) — Orchestrates the full optimization sequence: load the prior, blend with views via Black-Litterman, optimize weights, run risk checks, estimate transaction costs, and write target weights to the database. Also handles the "load views from DB" path when re-running an already-completed date.

---

### `src/execution/`

**`state.py`** — Reads and writes portfolio state (what you currently own). A key design decision: cash is stored as ticker `"CASH"` where 1 share = $1, making the portfolio a single consistent dictionary like `{"CASH": 180.15, "XLK": 1351.0, "XLF": 1874.0, ...}`. The first time this is called on an empty database, it returns `$1,000,000` in cash and zero shares of everything else.

**`orders.py`** — Translates target weights into concrete orders. For each ETF, computes `target_value = target_weight × portfolio_value`, compares to `current_value = shares_owned × price`, and generates a buy or sell order for the difference. Share counts are always rounded *down* (`math.floor`) so the system never tries to spend more than it has. Also contains `validate_orders_affordable` which checks whether available cash covers all the buys (after sell proceeds come in) and proportionally scales down buy quantities if not.

**`costs.py`** — Calculates transaction costs. Formula: `cost_USD = trade_value × (spread_bps + slippage_bps) / 10,000`. With spread=1 bp and slippage=2 bp, a $100,000 trade costs $30 in simulated fees. Skips trades below a 0.1% weight-change threshold (pure rounding noise).

**`fill_simulator.py`** — Simulates what happens when orders execute. For a buy: you pay `shares × price` plus transaction cost. For a sell: you receive `shares × price` minus cost. Writes one row to the `trades` table per filled order. Raises `NegativeCashError` if cash goes negative after fills (which should never happen under normal operation given `validate_orders_affordable`).

---

### `src/eval/`

**`attribution.py`** — Performance analysis tools. `compute_sector_contribution` estimates how much each sector contributed to the portfolio's total return (using average weight × sector return). `compute_cost_drag` reports what percentage of the portfolio's return was eaten by transaction costs. Always surfaces an `unexplained_pct` field acknowledging the approximation gap.

---

### `src/`

**`cache.py`** — SHA256 file cache for all LLM API calls. Before calling the Anthropic API, it serialises the entire call (model, system prompt, user message, tool definition) and computes a hash. If a file matching that hash already exists in `data/cache/`, it returns the cached response instantly (cost: $0, latency: <1ms). If not, it calls the API, saves the result, and logs the call to the `agent_calls` table. Without this, every historical backtest date would cost real money on every re-run.

**`pricing.py`** — Converts Anthropic token counts into dollar costs. Knows the per-token price for each model. Also provides a correction factor: tool schema serialization adds ~3,200 input tokens of fixed overhead per call, and structured tool call responses run 3–4× the naive token estimate.

**`config.py`** — `load_config(name)` reads a YAML file from `config/`, parses it into a typed Pydantic model, and returns it. All tunable parameters (thresholds, weights, model names, etc.) live in YAML rather than in source code.

**`exceptions.py`** — Custom exceptions: `TruncationError` (the LLM's response was cut off by the token limit) and `NegativeCashError` (a fill would make cash go below zero, indicating a budgeting bug).

**`weekly_run.py`** — The top-level weekly orchestrator. Runs all 9 steps in sequence: idempotency check → ingest fresh data → run agents → optimize → load positions → generate orders → simulate fills → write state → record snapshot. Separating this from `scripts/run_weekly.py` (the CLI wrapper) means tests can import and call it directly without going through `sys.argv`.

---

### `config/`

**`universe.yaml`** — Defines the 10-ETF investment universe plus SPY as benchmark. Every piece of code that needs the list of tickers reads this file rather than hardcoding.

**`agents.yaml`** — Maps each agent name to its Anthropic model string, prompt template file path, max token budget, and temperature. The macro agent uses `claude-sonnet-4-6` (more reasoning capacity); the news and events agents use `claude-haiku-4-5-20251001` (faster and cheaper for simpler tasks).

**`optimizer.yaml`** — All tunable optimizer parameters: tau, risk aversion, max position weight, volatility target, turnover penalty, transaction cost bps, prior lookback days, SPY sector weights, regime scale parameters, risk circuit breaker thresholds, and per-mode agent weights. See [Section 10](#10-every-configuration-parameter-what-it-controls-and-its-current-value) for the full table.

**`backtest.yaml`** — Backtest-specific settings: start date, end date, initial capital ($1,000,000), and rebalance frequency.

**`polymarket_markets.yaml`** — The 13 curated prediction-market questions, their Gamma API numeric IDs, confidence ratings, and sector impact mappings (e.g., "Fed rate cut by July" → positive for XLU and XLRE, negative for XLF). Refreshed quarterly.

**`sector_holdings.yaml`** — The top-10 company constituents for each of the 10 ETFs. Used by the news ingester to know which company tickers map to which sector.

---

## 4. What Each Agent Actually Does (Deep Dive)

### Agent 1 — NewsAgent (`agents/news_agent.py`)

**What data is pulled:** For each of the 10 ETFs, the agent queries the `news_raw` table for all articles from the trailing 7 days (`date - 7 days` to `date`), capped at 20 articles per sector. The `news_raw` table was populated by `ingest_alpha_vantage_news.py` (historical backfill) or `ingest_news.py` (weekly refresh). Each article has: `timestamp`, `ticker` (the company, e.g. "AAPL"), `sector` (the ETF, e.g. "XLK"), `title`, and `summary`.

**What goes to the LLM:** A JSON payload with two keys: `analysis_date` (e.g., `"2026-06-13"`) and `sectors` — a dict mapping each ETF ticker to a list of up to 20 articles, each with its date, company ticker, headline title, and up to 200 characters of summary. The system prompt (in `prompts/news_sentiment.txt`) instructs the model to act as a sector analyst and rate each sector. A forced tool call named `report_sector_sentiment` ensures the model can only respond in the prescribed JSON shape.

**What comes back:** A `NewsSignal` object with four fields:
- `sector_sentiments`: dict mapping each of the 10 ETF tickers to a float in `[-1.0, +1.0]`. Example from 2026-06-13: `XLK: +0.45`, `XLU: +0.55`, `XLV: +0.50`, `XLB: +0.20`.
- `sector_conviction`: dict mapping each ticker to a float in `[0.0, 1.0]`. The model is instructed to score ≤ 0.2 if fewer than 3 articles exist for a sector. Example: `XLK: 0.65`, `XLU: 0.75`, `XLV: 0.70`.
- `key_themes`: list of 3–5 short strings summarising the week's dominant narratives. Example: `"AI infrastructure capex cycle driving GPU and power demand"`, `"Geopolitical de-escalation (Iran peace signals) pressuring oil prices"`.
- `evidence`: for every sector with |sentiment| > 0.1, a citation of the specific headline that drove the score.

**What gets written to the database:** 10 rows in the `signals` table (one per sector). `agent_name = "sentiment"`, `target = "XLK"` (or whichever sector), `signal_value = sentiment_score`, `confidence = per_sector_conviction`. Also one row in `agent_calls` recording the model, token counts, cost, latency, and whether the cache was hit.

**Real example (2026-06-13):** The model read 200 articles (20 per sector, all sectors fully covered). It rated XLU (Utilities) the most bullish at +0.55 with conviction 0.75, likely driven by data center power demand stories. All 10 sectors were rated positive, which resulted in all Q values being positive — an unusual but coherent signal when macro also said `risk_on`.

---

### Agent 2 — MacroAgent (`agents/macro_agent.py`)

**What data is pulled:** Up to 400 days of all 7 FRED macro series from the `macro` table. Rather than passing all 400 raw data points to the LLM (wasteful and expensive), the agent pre-computes 16 derived features:
- VIX: current level, 30-day change, 90-day average.
- Yield curve (T10Y2Y): current spread, 90-day average, deviation from average.
- 10-year rate (DGS10): current level, 30-day change.
- CPI inflation: current index level, year-over-year percentage change (requires 365+ days of data).
- Unemployment (UNRATE): current rate, 3-month change.
- Jobless claims (ICSA): current week, 4-week average.
- USD index (DTWEXBGS): current level, 30-day change.

It also pulls up to 10 news articles each from XLF (financials) and XLI (industrials) as a macro news digest.

**What goes to the LLM:** A JSON payload with `analysis_date`, `derived_features` (the 16 numbers above), and `macro_news_digest` (the XLF+XLI articles). The model is more capable (`claude-sonnet-4-6`) because this task requires synthesising multiple indicators that can tell conflicting stories. The tool is called `report_macro_regime`.

**What comes back:** A `MacroRegimeSignal` with five fields:
- `reasoning`: A 150–300 word chain-of-thought explanation (stored but not directly used in the math).
- `regime`: one of `"risk_on"`, `"risk_off"`, or `"neutral"`.
- `rate_outlook`: one of `"rising"`, `"falling"`, or `"stable"`.
- `confidence`: float in `[0.0, 1.0]`. The model is instructed: 0.8+ only when all indicators agree; 0.3–0.5 for mixed signals.
- `rationale`: a 2–3 sentence human-readable summary.

**What gets written to the database:** Only 2 rows in `signals` (unlike the news agent's 10). Row 1: `target="macro_regime"`, `signal_value` converted to float (`risk_on → 1.0`, `neutral → 0.0`, `risk_off → −1.0`), `confidence = confidence`. Row 2: `target="rate_outlook"`, `signal_value` similarly encoded (`rising → 1.0`, `stable → 0.0`, `falling → −1.0`).

**Real example (2026-06-13):** VIX was 17.68 (below its 90-day average of 20.12 — a positive signal). The yield curve was +0.39% (positive — no recession warning). Unemployment was stable at 4.3%. The model said: `regime = "risk_on"`, `rate_outlook = "stable"`, `confidence = 0.72`. The key tension was CPI at 3.9% YoY — still above the Fed's 2% target — which capped confidence below the 0.8 threshold.

---

### Agent 3 — PolymarketAgent (`agents/polymarket_agent.py`)

#### The core idea: betting markets as a news source

Polymarket is a prediction market. Right now there are active questions like:
- "Will the Fed cut interest rates before July 2026?" (YES currently at 3% — the crowd thinks almost certainly not)
- "Will the US enter a recession in 2026?" (YES currently at 17% — the crowd thinks unlikely)
- "Will oil prices exceed $90/barrel by year end?" (YES at 42% — genuinely uncertain)

Each question has an implied probability — the current YES price expressed as a percentage. The crowd of bettors, staking real money, has collectively assigned a probability to each outcome.

The key insight is: **these probabilities are directly relevant to specific stock sectors.** Whether the Fed cuts rates matters a lot to utilities and real estate stocks (which behave like bonds and benefit from lower rates) but hurts bank stocks (which earn money on the spread between borrowing and lending rates — lower rates squeeze that spread). A recession being likely is bad for cyclical sectors like energy and industrials, but defensive sectors like healthcare and consumer staples hold up.

So if you know the crowd's best guess on 13 such questions, you can infer something about which sectors should be tilted up or down this week.

#### The 13 curated questions

The file `config/polymarket_markets.yaml` contains 13 hand-picked questions, chosen because they are macro-relevant and cover all 10 ETF sectors. For each question, the YAML also contains a pre-defined sector impact map, for example:

```
question: "Will the Fed cut rates before July 2026?"
sector_impacts:
  XLU: positive_if_yes     # utilities benefit from rate cuts
  XLRE: positive_if_yes    # real estate benefits from rate cuts
  XLF: negative_if_yes     # banks earn less when rates are cut
  XLK: neutral             # tech is not strongly rate-sensitive
```

This mapping is human-authored (not learned from data). It encodes economic intuition about which sectors are helped or hurt by each type of event.

#### What data is pulled

For each of the 13 questions, the agent queries `polymarket_raw` for:
- `current_prob` — the implied YES probability as of the analysis date (e.g., 0.03 = 3%)
- `prob_30d_ago` — what the probability was 30 days earlier (to see the trend)
- `volume_usd` — total dollars traded in this market (a proxy for how seriously to take it)
- `days_to_resolution` — when the question closes (a market resolving tomorrow is more actionable than one resolving in two years)

#### How a probability becomes a sector tilt — step by step

The LLM is given the question, the current probability, the trend, the volume, and the sector impact map. Its job is to produce a **sector tilt** for each of the 10 ETFs: a number from −1.0 to +1.0 saying whether upcoming events look good or bad for that sector.

The prompt instructs it to reason roughly like this (using "Will the Fed cut rates?" as the example):

**Step 1 — Centre the probability around neutral.**
A probability of 0.5 (50/50) means no information — ignore it. Anything above 0.5 leans YES; anything below leans NO. The "signal strength" is how far from 0.5 the probability is:
```
signal = current_prob − 0.5
       = 0.03 − 0.50 = −0.47   (strongly leans NO: rate cut is very unlikely)
```

**Step 2 — Apply the sector impact direction.**
The YAML says XLU is `positive_if_yes`. Since YES is very unlikely (signal is −0.47), that's bad for XLU:
```
raw_tilt_for_XLU = signal × direction
                 = −0.47 × (+1)  = −0.47
```
For XLF (`negative_if_yes`), the opposite: unlikely rate cut is good for banks:
```
raw_tilt_for_XLF = −0.47 × (−1) = +0.47
```

**Step 3 — Discount by how trustworthy the market is.**
Not all 13 markets deserve equal weight. The model discounts each signal by:
- **Volume** — a market with $500k in volume is more trustworthy than one with $5k. Low volume means the crowd is thin and the price can be moved by one big bettor.
- **Days to resolution** — a market resolving in 2 days is very actionable; one resolving in 18 months is too far away to affect this week's portfolio.
- **Confidence tier** — the YAML assigns each market a `high`/`medium`/`low` confidence rating based on how directly relevant it is to the sector universe. High-confidence markets discount less; low-confidence markets get heavily discounted.

After these discounts, the raw tilt of −0.47 might become −0.18 for XLU and +0.18 for XLF.

**Step 4 — Combine all 13 markets per sector.**
Multiple questions may affect the same sector. For XLU (utilities), there might be both a rate-cut question (bad for XLU if unlikely) and an infrastructure-spending question (good for XLU if likely). The model adds these up, clipping the final result to [−1.0, +1.0]. Tilts with absolute value below 0.05 snap to 0.0 (treated as noise).

**Step 5 — Use judgment, not just mechanics.**
The `judgments` field in the output is a scratchpad where the model is allowed to note cases where the mechanical rule would produce a wrong answer. For example: "Markets A and B are both about Fed rate cuts — they are nearly identical questions. I will only use market A (higher volume) and set market B's contribution to zero to avoid double-counting." This prevents correlated questions from amplifying the same signal artificially.

#### What comes back

A `PolymarketSignal` with six fields:
- `judgments`: the model's scratchpad for overrides (e.g., double-counting warnings).
- `implied_probs`: the current probabilities for all 13 markets (market_id → float). If the database has no data for a market (as happens on historical dates before Polymarket existed), the value is `None` and gets stripped before validation.
- `sector_tilts`: **the main output** — a dict mapping each ETF ticker to a single number in [−1.0, +1.0], representing the net effect of all 13 markets on that sector's outlook.
- `driving_events`: for each sector with a meaningful tilt (|tilt| ≥ 0.05), a human-readable explanation of which question drove the signal and why.
- `time_horizon`: `"short"`, `"medium"`, or `"long"` — the model's read of how near-term the signals are.
- `overall_confidence`: a single float in [0.0, 1.0] for the whole output. Reflects how much data was available and how consistent the markets' signals were.

#### What gets written to the database

10 rows in `signals`, one per sector. `agent_name = "events"`, `signal_value = sector_tilt`, `confidence = overall_confidence` (the same value for all 10 rows — Polymarket gives a portfolio-level confidence, not per-sector).

#### Why the same overall_confidence for all 10 sectors

Unlike the news agent (which reads different articles per sector and can say "XLK has 20 articles so I'm 0.75 confident, XLB has 2 articles so I'm 0.20 confident"), the Polymarket agent reads 13 global macroeconomic questions that affect all sectors simultaneously. The confidence in the data quality — how much volume, how many markets had live data — applies equally to all the outputs. So one number covers all 10 sectors.

#### Real example (2026-06-13)

All 13 markets had live data. Key probabilities:
- Fed rate cut by July: **3%** (very unlikely → NO is almost certain)
- US recession in 2026: **17%** (unlikely)
- Oil above $90/barrel: **42%** (genuinely uncertain, slight lean NO)

The model's key sector tilts:
- **XLU = −0.18** — rate cuts are extremely unlikely. Utilities are rate-sensitive (they pay steady dividends that become less attractive when rates stay high), so this is bearish for XLU.
- **XLRE = −0.15** — same logic as XLU. Real estate investment trusts borrow heavily and benefit from cheap money; no rate cuts means higher debt costs.
- **XLE = +0.18** — no recession expected (17% probability), which is good for energy demand. Oil consumption doesn't fall when the economy stays healthy.
- **XLF = +0.12** — no rate cuts is good for banks (higher rates = wider lending spreads = more profit).
- Overall confidence: **0.58** — markets were high-volume and consistent, but the mixed signals across 13 questions prevented full confidence.

---

## 5. How the Three Agents' Opinions Get Combined

The file `src/aggregator/views.py` runs after all three agents have written to the `signals` table. It produces two outputs for Black-Litterman: **Q** (the view on each sector's expected return) and **Ω** (how confident we are in each view).

### Step 1: Extract the macro regime multiplier

The macro agent's `macro_regime` signal is fetched (e.g., `+1.0` for `risk_on`). This feeds a simple formula:

```
regime_scale = regime_scale_intercept + regime_scale_slope × macro_regime_float
             = 0.75 + 0.25 × regime_float
```

So: `risk_off` (−1.0) → scale = **0.50**, `neutral` (0.0) → scale = **0.75**, `risk_on` (+1.0) → scale = **1.00**.

This means: in a `risk_off` environment, the other two agents' signals are automatically cut in half. The market is too fearful for sector tilts to be trusted. In `risk_on`, the full signal flows through.

### Step 2: Compute Q for each sector

For each of the 10 ETFs:

```
raw_signal = (news_weight × news_sentiment × news_conviction)
           + (poly_weight  × poly_tilt     × poly_confidence)

scaled_signal = raw_signal × regime_scale

Q = scaled_signal × max_excess_return_annual   (= ×0.05)
```

In **backtest mode**, `news_weight = 0.57`, `poly_weight = 0.00` (Polymarket has no historical data). In **live mode**, `news_weight = 0.40`, `poly_weight = 0.30`. The macro agent's weight (0.43 backtest / 0.30 live) only enters through the regime scale — it doesn't directly contribute a per-sector signal.

**Conviction controls the magnitude of the signal:** A news sentiment of +0.45 for XLK with conviction 0.65 contributes `0.57 × 0.45 × 0.65 = 0.167` to the raw signal. The same sentiment with conviction 0.20 (weak data) would contribute only `0.057` — less than a third as much.

**Worked example (XLK, 2026-06-13, live mode):**
- News: sentiment = +0.45, conviction = 0.65
- Poly: tilt = −0.08, confidence = 0.58
- Regime: risk_on, so scale = 1.00
- raw_signal = (0.40 × 0.45 × 0.65) + (0.30 × (−0.08) × 0.58) = 0.117 − 0.014 = 0.103
- scaled_signal = 0.103 × 1.00 = 0.103
- Q[XLK] = 0.103 × 0.05 = **+0.00515** (i.e., about 0.52% annualised excess return)

The actual value in the database for 2026-06-13: `Q[XLK] = +0.00515`. ✓

### Step 3: Compute Ω (view uncertainty) for each sector

Ω controls how hard the model "pulls" toward the agent views versus staying near the equilibrium. It's computed as:

```
agg_conviction = (news_weight × news_conviction + macro_weight × macro_confidence + poly_weight × poly_confidence) × regime_scale

Omega_entry = omega_base × 52 / max(agg_conviction, 0.01)
```

With `omega_base = 0.0001`: if `agg_conviction = 0.65` (high confidence), `Omega = 0.0001 × 52 / 0.65 = 0.008`. If `agg_conviction = 0.10` (low confidence), `Omega = 0.052` — six times larger. **A larger Ω means more uncertainty, meaning Black-Litterman will rely less on this view and stay closer to the equilibrium prior.** That is the mechanism by which a confident agent moves the portfolio more than an unconfident one.

The ×52 converts the weekly-calibrated `omega_base` to annual units, matching how Σ and π are expressed. This was a bug fix (ADR-020) — before the fix, Q and Ω were in mismatched units (weekly vs annual), causing the model to either ignore signals entirely or over-react to them.

---

## 6. How "Black-Litterman" Actually Works, No Formulas

### The problem it solves

Suppose the news agent says: "XLK looks great this week, I'd put 40% in it." If you just do that, you have a portfolio that's highly concentrated in tech and barely diversified. Meanwhile the market as a whole only puts 33% in tech — and the market reflects the collective wisdom of millions of traders. Your LLM agent read 200 articles; the market has priced in trillions of dollars of information. Who should you trust more?

Black-Litterman's answer: *trust both, weighted by how confident each is.*

### What the "prior" represents

If the agents had never said anything — if you started with a completely blank slate — the right answer is to hold the market portfolio (because any deviation from it implies you know something the market doesn't). The prior is the set of expected returns that would make the market weights optimal. For example, if XLK is 33% of the market, the prior return for XLK must be high enough to justify owning that much of it given its risk.

In this system, the prior (called π) is computed from SPY's published sector weights and the historical covariance of the 10 ETFs over the past 252 trading days (one year). As of 2026-06-13, the prior returns ranged roughly from −0.2% to +6.0% annually, with XLK (the largest sector at 32.9% of SPY) having one of the higher equilibrium returns.

### What the "posterior" represents

The posterior (called μ*) is the blended answer. It starts at the prior and gets nudged toward the agents' views — but only as much as the agents' conviction justifies. A highly confident agent (conviction = 0.80) pulls μ* most of the way toward its view. A barely-confident agent (conviction = 0.15) barely moves μ* away from the prior.

For 2026-06-13, the posterior returns ranged from about +0.1% to +4.2% annually. This is between the prior (−0.2% to +6.0%) and the agent views (Q ranging from +0.09% to +0.67%). The prior "dampened" the agents' views — which is exactly what you want, because you don't want a single week of bullish tech headlines to make you put all your money in XLK.

### Why a confident agent moves the portfolio more

Think of it this way: Ω is the "error bar" on the agent's view. A small error bar (small Ω = high conviction) means the agent is saying "I'm pretty sure about this." Black-Litterman responds by pulling μ* more strongly toward that view. A large error bar (large Ω = low conviction) means "I'm not very sure" — so Black-Litterman mostly ignores it and stays near the prior. The system is essentially a weighted average, where conviction = weight.

> **Optional math aside (skip if you want):**
> The formula is: μ* = M⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q] where M = (τΣ)⁻¹ + P'Ω⁻¹P. When Ω is small (high conviction), Ω⁻¹ is large, so Q pulls μ* strongly. When Ω is large (low conviction), Ω⁻¹ is small, so π dominates. τ = 0.05 controls how much the prior itself is trusted — small τ means the prior is very confident.

---

## 7. How Target Weights Get Decided (the Optimizer)

Given the posterior expected returns (μ*) and posterior covariance (Σ*) from Black-Litterman, the optimizer in `src/optimizer/portfolio.py` finds the weight vector that maximises:

```
expected return  −  (risk_aversion / 2) × portfolio_variance  −  turnover_penalty × sum_of_weight_changes
```

subject to four hard constraints (rules it can never break):

1. **Fully invested:** all weights sum to exactly 1.0 (100% of the portfolio is always deployed — no intentional cash sitting idle).
2. **Long-only:** every weight ≥ 0 (no short selling).
3. **Concentration cap:** no single sector ETF weight > 25%.
4. **Volatility ceiling:** the portfolio's annualised volatility ≤ 12%.

### What happens when the volatility ceiling can't be satisfied

During periods of extreme market stress (like April 2025, when VIX hit ~45 and all 10 ETFs were moving violently together), all possible sector portfolios may have volatility above 12%. The constraint becomes mathematically impossible — there is no allocation among these 10 assets that achieves ≤12% vol.

When this happens, the code tries CLARABEL (the primary solver), then SCS (the fallback). If both fail with the volatility constraint included, it drops the constraint and re-solves with only the other three rules. This is logged as `vol_constraint_status = "infeasible_relaxed"`. The optimizer doesn't give up and return the old weights unchanged — it finds the *best possible* portfolio under the remaining constraints, even if that means the resulting portfolio happens to be more volatile than usual.

### What the turnover penalty does

The optimizer objective subtracts `turnover_penalty × sum_of_|weight_changes|`. The coefficient `turnover_penalty = 0.002` (currently in `optimizer.yaml`) makes the optimizer reluctant to trade when the expected benefit is small.

**Concretely:** if moving 1% of the portfolio from XLE to XLK would improve expected annual return by 0.04%, the optimizer needs that gain to outweigh the penalty: `0.002 × 2 × 0.01 = 0.004%`. Since 0.04% > 0.004%, the trade happens. But if the benefit were only 0.002%, the penalty of 0.004% would exceed the gain — no trade.

**Why this matters:** Without this penalty, the optimizer would trade aggressively every week, racking up transaction costs and whipsaw-reacting to noisy weekly signals. The penalty ensures the system only trades when the signal is strong enough to justify the friction.

**What the number controls:** Larger values of `turnover_penalty` mean fewer, smaller trades. A value of 0.10 (the original setting before ADR-021) effectively implied a 10% one-way transaction cost per unit of weight moved — appropriate for illiquid private assets, completely wrong for 3-bps ETF trades. At 0.10, the optimizer refused to trade even when agents had a 4% cross-sectional return spread. The current 0.002 is provisional and will be calibrated in M5 via backtesting.

---

## 8. How a Trade Actually Happens (Execution)

### From target weight to share count

Suppose the optimizer says: target weight for XLK = 25.0% and the portfolio is worth $1,000,000.

1. **Target dollar value:** $1,000,000 × 0.25 = $250,000
2. **Current dollar value:** (shares owned) × (price). On the first ever run, 0 shares × $184.80 = $0.
3. **Delta:** $250,000 − $0 = $250,000 to buy
4. **Share count:** `floor($250,000 / $184.80)` = `floor(1352.6)` = **1352 shares**

The `floor` (always round down) is intentional: you never buy more shares than you can afford. The $0.60 of unspent "dust" stays in CASH.

### How transaction costs are calculated

For a buy of 1352 shares at $184.80 = $249,850 gross:

```
cost = $249,850 × (1 bps spread + 2 bps slippage) / 10,000
     = $249,850 × 0.0003
     = $74.96
```

So you'd end up paying $249,850 + $74.96 = $249,924.96 total, receiving 1352 shares. The $74.96 is recorded as `commission` in the `trades` table (slippage is already baked into this number, not double-counted as a separate field).

### How cash is tracked

Cash is stored as its own position: ticker `"CASH"`, where 1 share = $1. So `{"CASH": 180.15}` means you hold $180.15 in cash. This design means the entire portfolio is always represented as one consistent dictionary — you never need to separately track "ETF holdings" vs "leftover cash." Every fill adjusts the CASH position: buys reduce it, sells increase it.

### The affordability scaling fix

What if the optimizer wants to buy $300,000 of ETFs but you only have $280,000 in cash + sell proceeds? The `validate_orders_affordable` function detects this and proportionally scales down all buy quantities:

```
scale = funds_available / (total_buy_cost × (1 + cost_rate))
```

If scale = 0.93, every buy order gets its share count multiplied by 0.93 (floored to integer). No buy orders are dropped entirely (unless scale is so small a given order rounds to 0 shares). Sell orders are never touched. The system logs a WARNING with the scale factor. This is a graceful degradation — you get close to the target allocation rather than crashing.

---

## 9. A Single Real Walkthrough, Start to Finish

**Date: 2026-06-13 (the first ever live run)**

### Starting state

The database had no prior positions. `get_current_positions` returned the first-run fallback: `{"CASH": 1,000,000.0, "XLK": 0.0, ..., "XLU": 0.0}`. Portfolio value = $1,000,000.

### Step 1: NewsAgent

Read 200 articles (20 per sector, fully covered). All sectors rated positive that week. Key ratings:
- XLU: sentiment = **+0.55**, conviction = **0.75** (data center power demand stories)
- XLV: sentiment = **+0.50**, conviction = **0.70** (healthcare M&A + clinical wins)
- XLK: sentiment = **+0.45**, conviction = **0.65** (AI infrastructure capex)
- XLE: sentiment = **+0.30**, conviction = **0.55** (despite Iran peace signals, demand thesis held)
- XLB: sentiment = **+0.20**, conviction = **0.40** (weakest coverage, fewest clear stories)

Cost: $0 (cache hit). 10 signal rows written to DB.

### Step 2: MacroAgent

Derived features: VIX = 17.68, yield curve T10Y2Y = +0.39%, unemployment = 4.3% (stable), CPI YoY = 3.9%. Assessment: constructively **risk_on**, rate outlook = **stable**, confidence = **0.72**. The above-target CPI prevented the model from saying full confidence 0.8+. 2 signal rows written.

### Step 3: PolymarketAgent

13/13 markets had live data. No rate cuts expected (July cut probability 3%) → negative for rate-sensitive XLU and XLRE. No recession expected (17% probability) → positive for cyclicals XLE and XLF. The model weighed these against the high-volume markets. Key tilts:
- XLU: **−0.18**, XLY: **−0.15** (rate-cut skepticism hurt yield plays)
- XLE: **+0.18**, XLF: **+0.12** (no-recession base case)
- Overall confidence: **0.58** (markets were mixed). 10 signal rows written.

### Step 4: Aggregator (views.py)

Mode: live (news=0.40, macro=0.30 via regime only, poly=0.30). Regime = risk_on → scale = 1.00.

For **XLK** (worked example):
- raw = 0.40 × 0.45 × 0.65 + 0.30 × (−0.08) × 0.58 = 0.117 − 0.014 = **0.103**
- Q[XLK] = 0.103 × 1.00 × 0.05 = **+0.00515** (0.52% annualised excess return view)

For **XLU** (news very bullish, poly bearish):
- raw = 0.40 × 0.55 × 0.75 + 0.30 × (−0.18) × 0.58 = 0.165 − 0.031 = **0.134**
- Q[XLU] = 0.134 × 0.05 = **+0.00668** (0.67% — highest Q despite Polymarket being bearish)

Full Q vector (from DB):

| Sector | Q (annual) | Conviction |
|--------|-----------|------------|
| XLU    | +0.00668  | 0.690 |
| XLV    | +0.00665  | 0.670 |
| XLF    | +0.00524  | 0.630 |
| XLK    | +0.00515  | 0.650 |
| XLE    | +0.00487  | 0.610 |
| XLI    | +0.00246  | 0.590 |
| XLB    | +0.00230  | 0.550 |
| XLRE   | +0.00155  | 0.570 |
| XLY    | +0.00120  | 0.590 |
| XLP    | +0.00088  | 0.530 |

### Step 5: Black-Litterman (optimizer/equilibrium.py + black_litterman.py)

The prior (π) was computed from 252 days of price history and SPY sector weights. XLK (32.9% of SPY) had a high equilibrium return. The posterior μ* blended the prior with the views above (τ = 0.05, meaning the prior is trusted ~20× more than any single view). Result: posterior returns ranged from roughly **+0.1% to +4.2%** annually — XLK and XLV at the high end, XLP and XLB at the low end.

### Step 6: Optimizer (portfolio.py)

Maximised `μ* @ w − (2.5/2) × w'Σ*w − 0.002 × |w − w_prev|₁` subject to weights summing to 1, all weights ≥ 0, all weights ≤ 25%, and portfolio vol ≤ 12%.

Result:

| Sector | Target Weight | Note |
|--------|--------------|------|
| XLK    | **25.0%**    | At the 25% cap |
| XLF    | 10.0%        | |
| XLY    | 10.0%        | |
| XLV    | 10.0%        | |
| XLU    | 10.0%        | |
| XLE    | 10.0%        | |
| XLP    | 10.0%        | |
| XLI    |  9.7%        | Slightly below round number |
| XLRE   |  5.3%        | |
| XLB    |  0.0%        | Optimizer assigned zero |

Portfolio volatility at these weights: **9.55%** (well under the 12% ceiling — status = `"not_binding"`).

XLB got 0% despite a positive Q because its Q (+0.00230) was the lowest and the turnover penalty made it not worth buying.

### Step 7: Risk checks

All four checks passed:
- Drawdown circuit breaker: no prior portfolio history → automatically passed (need ≥2 snapshots).
- Realized vol: no prior history → automatically passed.
- Max position: highest weight = 25.0% (at the cap but not over).
- Max turnover: 0.30 (30% changed, within 50% limit).

No actions triggered. Final weights = optimizer output unchanged.

### Step 8: Order generation and fills

ETF prices (from most recent available date, 2026-06-12): XLK = $184.80, XLF = $53.34, XLV = $153.81, XLY = $116.60, XLU = $44.53, XLE = $57.55, XLB = $52.18, XLP = $85.82, XLRE = $45.36, XLI = $176.18.

Since all current positions were 0, every order was a buy. All sells completed before buys (per the "sells first" invariant). Key orders executed:

| Sector | Shares Bought | Price   | Dollar Value |
|--------|--------------|---------|-------------|
| XLK    | 1,351        | $184.80 | $249,685    |
| XLF    | 1,874        | $53.34  | $99,959     |
| XLU    | 2,245        | $44.53  | $99,950     |
| XLV    | 650          | $153.81 | $99,977     |
| XLE    | 1,737        | $57.55  | $99,954     |
| XLI    | 548          | $176.18 | $96,547     |
| XLP    | 1,165        | $85.82  | $99,980     |
| XLRE   | 1,177        | $45.36  | $53,389     |
| XLY    | 857          | $116.60 | $99,926     |
| XLB    | 1            | $52.18  | $52         |

Total transaction costs: ~$300 (10 sectors × ~$30 each at 3 bps on ~$100k).

### Step 9: Portfolio after rebalance

| Position | Shares  | Price   | Value      |
|----------|---------|---------|-----------|
| CASH     | 180.15  | $1.00   | $180       |
| XLK      | 1,351   | $184.80 | $249,685   |
| XLF      | 1,874   | $53.34  | $99,959    |
| XLU      | 2,245   | $44.53  | $99,950    |
| XLV      | 650     | $153.81 | $99,977    |
| ... (all others) |  |        | ~$450,000  |

**Total portfolio value: $999,609** (started at $1,000,000; lost $391 to transaction costs and rounding). Written to `portfolio_snapshot` table. The system is now deployed — subsequent weekly runs will generate smaller incremental rebalances rather than starting from all-cash.

---

## 10. Every Configuration Parameter, What It Controls, and Its Current Value

### `config/optimizer.yaml`

| Parameter | Current Value | What It Controls | Effect of Increasing | Effect of Decreasing |
|-----------|--------------|------------------|---------------------|---------------------|
| `tau` | 0.05 | How much the prior is trusted relative to the views. Small tau = prior is very confident, agents move it less. | Prior dominates even more; agents nearly ignored | Agents can move portfolio further from market weights |
| `risk_aversion` | 2.5 | How much the optimizer penalises volatility. Higher = prefers lower-risk portfolios. | More concentrated in low-vol sectors | More willing to hold high-vol sectors for higher expected return |
| `max_position_weight` | 0.25 | Maximum fraction of portfolio in any single ETF. | Allows more concentration | Forces more diversification |
| `vol_target` | 0.12 | Annualised portfolio volatility ceiling (12%). | Allows riskier portfolios | Forces more conservative weights (or triggers infeasible_relaxed more often) |
| `turnover_penalty` | 0.002 | Coefficient γ penalising weight changes. Controls how reluctant the system is to trade. **Provisional — must be calibrated in M5.** Was 0.10 before ADR-021 (too high: implied 10% one-way cost, causing zero trades). Now 0.002, which is ~3× actual 3bps ETF cost, providing a noise buffer. | Fewer, smaller trades; system anchors closer to previous weights | More aggressive trading; risk of overreacting to noisy weekly signals |
| `transaction_cost_bps` → `spread_bps` | 1.0 | Half-spread estimate in basis points (one side of the bid-ask spread). | Higher cost charged per trade | Lower cost charged |
| `transaction_cost_bps` → `slippage_bps` | 2.0 | Market impact estimate. Together with spread, total one-way cost = 3 bps. | Higher cost charged per trade | Lower cost charged |
| `transaction_costs.min_trade_threshold` | 0.001 | Minimum weight change to generate a trade and charge a cost (0.1%). Changes below this are treated as rounding noise. | Fewer very small trades | More micro-trades, more cost noise |
| `prior.lookback_days` | 252 | How many trading days of price history used to estimate the covariance matrix (one year). | More history, more stable covariance but slower to react to volatility regime changes | Less history, more reactive covariance |
| `aggregator.max_excess_return_annual` | 0.05 | Maps a ±1.0 sentiment signal to ±5% annual excess return. This is the scale of Q. Was wrong before ADR-020 when Q was in weekly units (÷52 too small, making views nearly invisible). | Agent views are taken more literally (higher Q magnitudes) | Agents have less ability to move expected returns |
| `aggregator.omega_base` | 0.0001 | Base view variance at unit conviction. Together with the ×52 annualisation and conviction scaling, determines how much each view can move μ* away from the prior. Fixed after ADR-020 to match annual units. | More uncertainty → prior dominates more | Less uncertainty → agent views dominate |
| `aggregator.regime_scale_intercept` | 0.75 | The multiplier applied to signals when macro regime is neutral (0). | Neutral regime = higher throughput of signals | Neutral regime = signals more suppressed |
| `aggregator.regime_scale_slope` | 0.25 | How much the scale changes per unit of regime. With intercept 0.75 and slope 0.25: risk_off → 0.50, neutral → 0.75, risk_on → 1.00. | Larger difference between risk_on and risk_off behaviour | All regimes treated more similarly |
| `risk.max_single_rebalance_turnover` | 0.50 | Maximum L1 turnover (sum of |weight changes|) in a single rebalance. If exceeded, blend 50/50 with previous weights. | Allows more dramatic weekly changes | More conservative rebalances |
| `risk.max_drawdown_threshold` | 0.15 | Rolling peak-to-trough drawdown threshold (15%). If exceeded, halt rebalancing entirely. | Higher drawdown required to halt | Rebalances halt earlier in drawdowns |
| `risk.drawdown_lookback_days` | 20 | How many days of portfolio history used to compute rolling drawdown and realized vol. | More stable estimates (slower to react) | Reacts faster to recent stress |
| `risk.vol_breach_multiplier` | 1.50 | Realized vol must exceed `vol_target × 1.50` = 18% to trigger deleveraging. | Deleveraging only on extreme vol spikes | Deleveraging triggered more frequently |
| `risk.vol_deleveraging_blend` | 0.20 | When vol breach fires, blend 20% toward equal weights (10% each). | Stronger deleveraging on vol spike | Milder deleveraging |
| `aggregator_weights.backtest.news` | 0.57 | News agent weight in backtest mode. Higher because Polymarket has no historical data. |  |  |
| `aggregator_weights.backtest.macro` | 0.43 | Macro agent weight in backtest mode (enters only via regime scale). |  |  |
| `aggregator_weights.backtest.polymarket` | 0.00 | Polymarket excluded from backtest (no historical data before ~2025). |  |  |
| `aggregator_weights.live.news` | 0.40 | News weight in live mode (lower because Polymarket now contributes). |  |  |
| `aggregator_weights.live.macro` | 0.30 | Macro weight in live mode. |  |  |
| `aggregator_weights.live.polymarket` | 0.30 | Polymarket weight in live mode. |  |  |

### `config/backtest.yaml`

| Parameter | Current Value | What It Controls |
|-----------|--------------|-----------------|
| `start_date` | 2023-06-01 | First date in the backtest run |
| `end_date` | 2024-06-01 | Last date in the backtest run |
| `initial_capital` | 1,000,000 | Starting cash in USD |
| `rebalance_frequency` | weekly | How often to rebalance |

### `config/agents.yaml`

| Parameter | Current Value | What It Controls |
|-----------|--------------|-----------------|
| `sentiment.model` | claude-haiku-4-5-20251001 | LLM used for news sentiment (fast + cheap) |
| `macro.model` | claude-sonnet-4-6 | LLM used for macro regime (more reasoning capacity) |
| `events.model` | claude-haiku-4-5-20251001 | LLM used for Polymarket events |
| `sentiment.max_tokens` | 4096 | Max tokens in news agent response |
| `macro.max_tokens` | 2048 | Max tokens in macro agent response |
| `temperature` | 0.0 (all agents) | Determinism — 0.0 means identical inputs always produce identical outputs (important for cache hits to work correctly) |

---

## 11. Known Limitations and Open Questions

**Turnover penalty is not yet calibrated (ADR-021).** The current value of 0.002 was set to be "not obviously broken" (it produces trades when signals are strong) but has never been tested across a full backtest. The M5 backtester needs to sweep γ ∈ {0.001, 0.002, 0.005, 0.01} and measure the effect on weekly turnover, transaction cost drag, and net Sharpe ratio before any value can be called final.

**Polymarket has no historical data before ~late 2024.** The 13 curated markets were all created recently. Any backtest run before those dates will show `markets_with_data = 0/13`, forcing the Polymarket agent to return a cached zero-signal response. In backtest mode this is handled by setting polymarket weight to 0.00, but the missing data gap means live-mode performance cannot be directly compared to backtest-mode performance.

**News data coverage is uneven across time.** Alpha Vantage's free tier allows 25 API calls per day, with one ticker per call. The 100-ticker universe (10 sectors × ~10 tickers each) takes 4 days to backfill. If the backfill was interrupted or some calls failed silently (AV doesn't error on empty results), certain sectors may have fewer articles for certain historical periods, making those backtested signals less reliable.

**The Q-to-return mapping is linear and capped at ±5%.** A sentiment of +1.0 always maps to a +5% annualised excess return view, regardless of what kind of market regime we're in. In a high-inflation environment, 5% excess return might be trivially achievable; in a flat market, it might be wildly optimistic. There's no dynamic scaling based on market conditions.

**The macro agent has no memory of previous regimes.** It classifies the regime freshly from current indicators each week, without knowing that three weeks ago it said `risk_off`. It cannot detect a regime transition (going from `risk_off` to `neutral` is a potentially more important signal than being `neutral` when you've always been neutral). This is a fundamental limitation of the stateless prompt-in, response-out design.

**Agent lookback parameters are hardcoded in source files.** `_MAX_ARTICLES_PER_SECTOR = 20`, `_NEWS_LOOKBACK_DAYS = 7`, `_MACRO_LOOKBACK_DAYS = 400`, `_MAX_NEWS_ARTICLES = 10` are defined as module-level constants rather than read from `agents.yaml`. This violates the project's "all tunable parameters in YAML" rule and makes it harder to experiment with different lookback windows during backtesting. Noted as an open item in `CLAUDE.md`.

**The drawdown and realized-vol circuit breakers have never been triggered.** The portfolio only has data for one date (2026-06-13), which is not enough history for the drawdown check (needs ≥2 snapshots) or the realized-vol check (needs ≥3). These will begin functioning once the M5 backtester runs across a multi-week period. Their thresholds (15% drawdown, 1.5× vol target) have not been validated against historical stress periods.

**No live trading execution yet.** The "execution" layer only simulates fills — it does not actually send orders to a broker or paper-trading API. The system is designed for AWS deployment with a real broker in a later phase, but that infrastructure does not exist yet.

**SPY sector weights need quarterly refreshes.** The weights in `optimizer.yaml` under `market_cap_weights` reflect SSGA's published SPY factsheet from 2026-06-21. SPY rebalances quarterly, so these weights become stale. XLV was discovered to be 2.73 percentage points wrong on 2026-06-21 (ADR-019) because an earlier version used the wrong data source. A reminder to refresh in September 2026 is noted in `decisions.md`.

---

*File: `docs/ARCHITECTURE_EXPLAINED.md`*
*Length: approximately 6,800 words across 11 sections.*
