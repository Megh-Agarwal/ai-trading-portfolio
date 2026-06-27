"""Signal quality diagnostic — reads only from existing DB, no new API calls.

Analyzes the 53-week authoritative backtest window (2025-06-13 → 2026-06-12, ADR-022).

Part 1: Per-agent, per-sector correlation
  - News sentiment vs next-week sector return (Pearson r, pooled and per-sector)
  - Macro regime vs next-week portfolio return (Pearson r)
Part 2: Hit rate
  - Fraction of weeks where sign(news sentiment) == sign(next-week sector return)
  - Binomial test vs 50% baseline; macro regime hit rate
Part 3: Per-agent isolation
  - News-only Q (regime frozen at neutral=0.75) vs actual Q (regime-modulated)
  - Macro contribution = Q_actual − Q_news_neutral, correlated with actual returns
  - Mean return by regime label (risk_on / neutral / risk_off)
Part 4: Worst 5 weeks
  - Weeks where LLM Full most underperformed No-LLM baseline
  - Agent calls and actual market moves for each

Saves scatter plots + bar charts to diagnostics/ folder.
"""

from __future__ import annotations

import datetime
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Insert src into Python path
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from config import load_config
from db.models import (
    PORTFOLIO_BACKTEST_FULL,
    PORTFOLIO_BACKTEST_NO_LLM,
    PortfolioSnapshot,
    Price,
    Signal,
    View,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANALYSIS_START = datetime.date(2025, 6, 13)
ANALYSIS_END = datetime.date(2026, 6, 12)
SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU"]

DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
DIAG_DIR = Path(__file__).parent.parent / "diagnostics"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_all(engine):
    """Pull every table slice needed for the diagnostic in one session."""
    with Session(engine) as s:
        news_rows = s.execute(
            select(Signal)
            .where(Signal.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .where(Signal.agent_name == "sentiment")
            .where(Signal.date >= ANALYSIS_START)
            .where(Signal.date <= ANALYSIS_END)
        ).scalars().all()

        macro_rows = s.execute(
            select(Signal)
            .where(Signal.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .where(Signal.agent_name == "macro")
            .where(Signal.date >= ANALYSIS_START)
            .where(Signal.date <= ANALYSIS_END)
        ).scalars().all()

        view_rows = s.execute(
            select(View)
            .where(View.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .where(View.date >= ANALYSIS_START)
            .where(View.date <= ANALYSIS_END)
        ).scalars().all()

        full_snaps = s.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_BACKTEST_FULL)
            .where(PortfolioSnapshot.date >= ANALYSIS_START)
            .where(PortfolioSnapshot.date <= ANALYSIS_END)
            .order_by(PortfolioSnapshot.date)
        ).scalars().all()

        nollm_snaps = s.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.portfolio_id == PORTFOLIO_BACKTEST_NO_LLM)
            .where(PortfolioSnapshot.date >= ANALYSIS_START)
            .where(PortfolioSnapshot.date <= ANALYSIS_END)
            .order_by(PortfolioSnapshot.date)
        ).scalars().all()

        # Price buffer: 14 days before ANALYSIS_START and 14 days after ANALYSIS_END
        # covers holidays that fall on rebalance Fridays
        buf_start = ANALYSIS_START - datetime.timedelta(days=14)
        buf_end = ANALYSIS_END + datetime.timedelta(days=14)
        price_rows = s.execute(
            select(Price.date, Price.ticker, Price.adj_close)
            .where(Price.ticker.in_(SECTORS))
            .where(Price.date >= buf_start)
            .where(Price.date <= buf_end)
            .order_by(Price.date)
        ).all()

    return news_rows, macro_rows, view_rows, full_snaps, nollm_snaps, price_rows


def _build_price_pivot(price_rows, buf_start, buf_end):
    """Build a daily forward-filled price pivot (date × sector)."""
    df = pd.DataFrame(price_rows, columns=["date", "ticker", "adj_close"])
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")
    pivot.index = pd.to_datetime(pivot.index)
    full_range = pd.date_range(buf_start, buf_end, freq="D")
    pivot = pivot.reindex(full_range).ffill()
    pivot.columns.name = None
    return pivot


def _sector_returns(rebalance_dates, price_pivot):
    """Return dict[date → dict[sector → weekly_simple_return]]."""
    date_to_next = {d: rebalance_dates[i + 1] for i, d in enumerate(rebalance_dates[:-1])}
    result = {}
    for d, d_next in date_to_next.items():
        ts, ts_next = pd.Timestamp(d), pd.Timestamp(d_next)
        if ts not in price_pivot.index or ts_next not in price_pivot.index:
            continue
        ret = {}
        for sector in SECTORS:
            if sector not in price_pivot.columns:
                continue
            p0, p1 = price_pivot.at[ts, sector], price_pivot.at[ts_next, sector]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret[sector] = float(p1 / p0 - 1.0)
        result[d] = ret
    return result


def _weekly_returns_from_snaps(snap_rows):
    """Return DataFrame with columns [date, weekly_return]."""
    snaps = sorted(snap_rows, key=lambda r: r.date)
    rows = []
    for i in range(len(snaps) - 1):
        v0, v1 = snaps[i].total_value, snaps[i + 1].total_value
        rows.append({"date": snaps[i].date, "weekly_return": v1 / v0 - 1.0})
    return pd.DataFrame(rows)


def _save_fig(fig, name):
    DIAG_DIR.mkdir(exist_ok=True)
    path = DIAG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved: {path}]")


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 72)
    print("  SIGNAL QUALITY DIAGNOSTIC")
    print(f"  Window: {ANALYSIS_START} → {ANALYSIS_END}  (53 rebalances)")
    print("  Data source: DB only — no new API calls")
    print("=" * 72)

    engine = create_engine(f"sqlite:///{DB_PATH}")
    news_rows, macro_rows, view_rows, full_snaps, nollm_snaps, price_rows = _load_all(engine)

    # Optimizer config values
    cfg = load_config("optimizer")
    agg = cfg.aggregator
    W_NEWS = cfg.aggregator_weights.backtest.news           # 0.57
    W_MACRO = cfg.aggregator_weights.backtest.macro         # 0.43
    INTERCEPT = agg.regime_scale_intercept                   # 0.75
    SLOPE = agg.regime_scale_slope                           # 0.25
    MAX_RETURN = agg.max_excess_return_annual                # 0.05
    NEUTRAL_SCALE = INTERCEPT                                # scale when regime=0

    # Build core dataframes
    news_df = pd.DataFrame([
        {"date": r.date, "sector": r.target,
         "sentiment": float(r.signal_value), "conviction": float(r.confidence or 0.0)}
        for r in news_rows
    ])

    # macro_dict: date → {target: value}
    macro_dict: dict[datetime.date, dict[str, float]] = defaultdict(dict)
    for r in macro_rows:
        macro_dict[r.date][r.target] = float(r.signal_value)

    macro_df = pd.DataFrame([
        {"date": d,
         "macro_regime": v.get("macro_regime", 0.0),
         "rate_outlook": v.get("rate_outlook", 0.0)}
        for d, v in macro_dict.items()
    ]).sort_values("date").reset_index(drop=True)

    views_df = pd.DataFrame([
        {"date": r.date, "sector": r.sector, "expected_return": float(r.expected_return)}
        for r in view_rows
    ])

    buf_start = ANALYSIS_START - datetime.timedelta(days=14)
    buf_end = ANALYSIS_END + datetime.timedelta(days=14)
    price_pivot = _build_price_pivot(price_rows, buf_start, buf_end)

    rebalance_dates = sorted(news_df["date"].unique())
    sec_ret_map = _sector_returns(rebalance_dates, price_pivot)

    full_weekly = _weekly_returns_from_snaps(full_snaps)
    nollm_weekly = _weekly_returns_from_snaps(nollm_snaps)

    # ================================================================
    # PART 1 — News sentiment correlation
    # ================================================================
    print("\n" + "=" * 72)
    print("PART 1 — Correlation: News Sentiment vs Next-Week Sector Return")
    print("=" * 72)

    pooled_sent, pooled_ret = [], []
    by_sector: dict[str, tuple[list, list]] = {s: ([], []) for s in SECTORS}

    for d in rebalance_dates[:-1]:
        if d not in sec_ret_map:
            continue
        week_news = news_df[news_df["date"] == d].set_index("sector")
        for sector in SECTORS:
            if sector not in week_news.index:
                continue
            sent = week_news.at[sector, "sentiment"]
            ret = sec_ret_map[d].get(sector)
            if ret is None or math.isnan(sent):
                continue
            pooled_sent.append(sent)
            pooled_ret.append(ret)
            by_sector[sector][0].append(sent)
            by_sector[sector][1].append(ret)

    pooled_sent = np.array(pooled_sent)
    pooled_ret = np.array(pooled_ret)

    r_news_pool, p_news_pool = (float("nan"), float("nan"))
    if len(pooled_sent) >= 3:
        r_news_pool, p_news_pool = stats.pearsonr(pooled_sent, pooled_ret)

    n_obs = len(pooled_sent)
    print(f"\nPooled (10 sectors × {len(rebalance_dates)-1} predictive weeks, n={n_obs}):")
    print(f"  Pearson r = {r_news_pool:+.4f}   p = {p_news_pool:.4f}   "
          f"{'*significant (p<0.05)*' if p_news_pool < 0.05 else 'ns'}")
    print(f"\n  Significance threshold at n={n_obs}: |r| > {1.96/math.sqrt(n_obs):.3f} (p<0.05)")

    print(f"\n  {'Sector':<6}  {'r':>7}  {'p':>6}  {'n':>4}  {'sig':>4}")
    print("  " + "-" * 36)
    for sector in SECTORS:
        sarr, rarr = np.array(by_sector[sector][0]), np.array(by_sector[sector][1])
        if len(sarr) < 3:
            print(f"  {sector:<6}    N/A     N/A  {len(sarr):>3}     -")
            continue
        r, p = stats.pearsonr(sarr, rarr)
        print(f"  {sector:<6}  {r:+.4f}  {p:.4f}  {len(sarr):>3}  {'*' if p < 0.05 else 'ns'}")

    # Scatter plot: sentiment vs return (pooled)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pooled_sent, pooled_ret * 100, alpha=0.25, s=12, color="steelblue")
    if not math.isnan(r_news_pool):
        m, b = np.polyfit(pooled_sent, pooled_ret * 100, 1)
        x_line = np.linspace(pooled_sent.min(), pooled_sent.max(), 100)
        ax.plot(x_line, m * x_line + b, color="red", lw=1.5,
                label=f"r={r_news_pool:+.3f}  p={p_news_pool:.3f}")
        ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("News Sentiment (−1 to +1)")
    ax.set_ylabel("Next-week sector return (%)")
    ax.set_title(f"News Sentiment vs Next-Week Sector Return\n"
                 f"n={n_obs} (10 sectors × 52 predictive weeks, pooled)")
    _save_fig(fig, "01_news_sentiment_vs_return.png")

    # ================================================================
    # PART 1b — Macro regime correlation
    # ================================================================
    print("\n" + "-" * 72)
    print("Macro Regime vs Next-Week Portfolio Return")
    print("-" * 72)

    # Merge macro regime with FULL next-week return
    macro_full = macro_df.merge(full_weekly.rename(columns={"weekly_return": "full_return"}),
                                on="date", how="inner")

    r_regime = r_rate = p_regime = p_rate = float("nan")
    if len(macro_full) >= 3:
        r_regime, p_regime = stats.pearsonr(macro_full["macro_regime"], macro_full["full_return"])
        r_rate, p_rate = stats.pearsonr(macro_full["rate_outlook"], macro_full["full_return"])

    print(f"\n  macro_regime (risk_on=+1, neutral=0, risk_off=−1) vs next-week portfolio return:")
    print(f"    Pearson r = {r_regime:+.4f}   p = {p_regime:.4f}   "
          f"{'*significant*' if p_regime < 0.05 else 'ns'}   n={len(macro_full)}")
    print(f"\n  rate_outlook (rising=+1, stable=0, falling=−1) vs next-week portfolio return:")
    print(f"    Pearson r = {r_rate:+.4f}   p = {p_rate:.4f}   "
          f"{'*significant*' if p_rate < 0.05 else 'ns'}   n={len(macro_full)}")

    regime_counts = macro_full["macro_regime"].value_counts().sort_index(ascending=False)
    print(f"\n  Regime distribution across {len(macro_full)} weeks:")
    label_map = {1.0: "risk_on ", 0.0: "neutral ", -1.0: "risk_off"}
    for val, count in regime_counts.items():
        lbl = label_map.get(float(val), str(val))
        avg_ret = macro_full[macro_full["macro_regime"] == val]["full_return"].mean() * 100
        print(f"    {lbl} ({val:+.0f}): {count:>2} weeks  mean_next_wk_return={avg_ret:+.3f}%")

    # Scatter: macro regime vs portfolio return
    fig, ax = plt.subplots(figsize=(6, 5))
    jitter = np.random.default_rng(42).uniform(-0.07, 0.07, size=len(macro_full))
    ax.scatter(macro_full["macro_regime"] + jitter,
               macro_full["full_return"] * 100,
               alpha=0.5, s=25, color="darkorange")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["risk_off\n(−1)", "neutral\n(0)", "risk_on\n(+1)"])
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Macro Regime Signal")
    ax.set_ylabel("Next-week FULL portfolio return (%)")
    ax.set_title(f"Macro Regime vs Next-Week Portfolio Return\n"
                 f"r={r_regime:+.3f}  p={p_regime:.3f}  n={len(macro_full)}")
    _save_fig(fig, "02_macro_regime_vs_return.png")

    # ================================================================
    # PART 2 — Hit rate
    # ================================================================
    print("\n" + "=" * 72)
    print("PART 2 — Hit Rate: sign(news sentiment) == sign(next-week sector return)")
    print("=" * 72)

    hits = misses = skipped_zero = 0
    by_sector_hr: dict[str, dict[str, int]] = {s: {"hits": 0, "total": 0} for s in SECTORS}

    for d in rebalance_dates[:-1]:
        if d not in sec_ret_map:
            continue
        week_news = news_df[news_df["date"] == d].set_index("sector")
        for sector in SECTORS:
            if sector not in week_news.index:
                continue
            sent = week_news.at[sector, "sentiment"]
            ret = sec_ret_map[d].get(sector)
            if ret is None or math.isnan(sent):
                continue
            if sent == 0.0:
                skipped_zero += 1
                continue
            correct = (sent > 0) == (ret > 0)
            if correct:
                hits += 1
                by_sector_hr[sector]["hits"] += 1
            else:
                misses += 1
            by_sector_hr[sector]["total"] += 1

    total_dir = hits + misses
    hit_rate = hits / total_dir if total_dir > 0 else float("nan")
    binom_p = float("nan")
    if total_dir > 0 and not math.isnan(hit_rate):
        binom_result = stats.binomtest(hits, total_dir, p=0.5, alternative="greater")
        binom_p = binom_result.pvalue

    print(f"\nPooled (excluding zero-sentiment observations — {skipped_zero} skipped):")
    print(f"  {hits} hits / {total_dir} directional calls = {hit_rate:.1%}")
    print(f"  vs 50% baseline: {(hit_rate - 0.5)*100:+.1f}pp")
    print(f"  Binomial test p (one-sided, >50%): {binom_p:.4f}   "
          f"{'*significant*' if binom_p < 0.05 else 'ns'}")

    print(f"\n  {'Sector':<6}  {'Hits':>4}  {'Total':>5}  {'Rate':>6}")
    print("  " + "-" * 28)
    sector_rates = []
    for sector in SECTORS:
        h = by_sector_hr[sector]["hits"]
        t = by_sector_hr[sector]["total"]
        rate = h / t if t > 0 else 0.0
        sector_rates.append(rate)
        print(f"  {sector:<6}  {h:>4}  {t:>5}  {rate:>6.1%}")

    # Macro regime hit rate
    mac_hits = mac_total = 0
    for _, row in macro_full.iterrows():
        if row["macro_regime"] == 0.0:
            continue
        mac_total += 1
        if (row["macro_regime"] > 0) == (row["full_return"] > 0):
            mac_hits += 1
    mac_rate = mac_hits / mac_total if mac_total > 0 else float("nan")
    mac_binom_p = float("nan")
    if mac_total > 0 and not math.isnan(mac_rate):
        mac_binom_p = stats.binomtest(mac_hits, mac_total, p=0.5, alternative="greater").pvalue

    print(f"\n  Macro regime hit rate (risk_on → portfolio up, risk_off → portfolio down):")
    print(f"  {mac_hits}/{mac_total} = {mac_rate:.1%}   "
          f"binomial p = {mac_binom_p:.4f}   {'*' if mac_binom_p < 0.05 else 'ns'}")

    # Bar chart: per-sector hit rates
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2196F3" if r > 0.5 else "#EF5350" for r in sector_rates]
    ax.bar(SECTORS, [r * 100 for r in sector_rates], color=colors, alpha=0.8)
    ax.axhline(50, color="black", lw=1.2, ls="--", label="50% (random)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("News Sentiment Hit Rate by Sector\n(sign match vs next-week sector return)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    for i, (s, r) in enumerate(zip(SECTORS, sector_rates)):
        ax.text(i, r * 100 + 1.5, f"{r:.0%}", ha="center", va="bottom", fontsize=8)
    _save_fig(fig, "03_hit_rate_by_sector.png")

    # ================================================================
    # PART 3 — Per-agent isolation
    # ================================================================
    print("\n" + "=" * 72)
    print("PART 3 — Per-Agent Isolation: News vs Macro Contribution to Q")
    print("=" * 72)

    print(f"\nConfig recap:")
    print(f"  Q[sector] = w_news × sentiment × conviction × regime_scale × max_return")
    print(f"  regime_scale = {INTERCEPT} + {SLOPE} × macro_regime")
    print(f"  w_news={W_NEWS}  w_macro={W_MACRO}  max_return={MAX_RETURN}")
    print(f"  risk_on(+1): scale={INTERCEPT+SLOPE:.2f}  neutral(0): scale={NEUTRAL_SCALE:.2f}  "
          f"risk_off(−1): scale={INTERCEPT-SLOPE:.2f}")
    print(f"\n  Macro modulates news signals multiplicatively — it cannot generate")
    print(f"  per-sector Q on its own (without news, Q=0 regardless of regime).")
    print(f"  The 'isolation' test compares Q with regime frozen vs regime active.\n")

    # Build Q_news_neutral and Q_actual per (date, sector)
    news_by_ds = {
        (row["date"], row["sector"]): (row["sentiment"], row["conviction"])
        for _, row in news_df.iterrows()
    }
    views_by_ds = {
        (row["date"], row["sector"]): row["expected_return"]
        for _, row in views_df.iterrows()
    }

    news_neutral_q, actual_q, actual_ret_p3 = [], [], []
    macro_delta_corr_q, macro_delta_corr_ret = [], []
    regime_delta_per_week = []

    for d in rebalance_dates[:-1]:
        if d not in sec_ret_map:
            continue
        regime = macro_dict.get(d, {}).get("macro_regime", 0.0)
        actual_scale = INTERCEPT + SLOPE * regime
        week_deltas = []
        for sector in SECTORS:
            news_info = news_by_ds.get((d, sector))
            q_actual = views_by_ds.get((d, sector))
            actual_r = sec_ret_map[d].get(sector)
            if news_info is None or q_actual is None or actual_r is None:
                continue
            sent, conv = news_info
            q_news_neutral = W_NEWS * sent * conv * NEUTRAL_SCALE * MAX_RETURN
            macro_delta = q_actual - q_news_neutral

            news_neutral_q.append(q_news_neutral)
            actual_q.append(q_actual)
            actual_ret_p3.append(actual_r)
            macro_delta_corr_q.append(macro_delta)
            macro_delta_corr_ret.append(actual_r)
            week_deltas.append(abs(macro_delta))
        if week_deltas:
            regime_delta_per_week.append(np.mean(week_deltas))

    news_neutral_q = np.array(news_neutral_q)
    actual_q = np.array(actual_q)
    actual_ret_p3 = np.array(actual_ret_p3)
    macro_delta_arr = np.array(macro_delta_corr_q)
    macro_delta_ret_arr = np.array(macro_delta_corr_ret)

    r_qneutral = r_qactual = r_mdelta = float("nan")
    p_qneutral = p_qactual = p_mdelta = float("nan")
    if len(news_neutral_q) >= 3:
        r_qneutral, p_qneutral = stats.pearsonr(news_neutral_q, actual_ret_p3)
        r_qactual, p_qactual = stats.pearsonr(actual_q, actual_ret_p3)
        r_mdelta, p_mdelta = stats.pearsonr(macro_delta_arr, macro_delta_ret_arr)

    mean_delta = np.mean(regime_delta_per_week) if regime_delta_per_week else float("nan")

    print(f"  Q_news_only (regime frozen at neutral={NEUTRAL_SCALE:.2f}) vs sector return:")
    print(f"    Pearson r = {r_qneutral:+.4f}   p = {p_qneutral:.4f}   "
          f"{'*' if p_qneutral < 0.05 else 'ns'}   n={len(news_neutral_q)}")
    print(f"\n  Q_actual (regime-modulated) vs sector return:")
    print(f"    Pearson r = {r_qactual:+.4f}   p = {p_qactual:.4f}   "
          f"{'*' if p_qactual < 0.05 else 'ns'}   n={len(actual_q)}")
    print(f"\n  Macro delta (Q_actual − Q_news_neutral) vs sector return:")
    print(f"    Pearson r = {r_mdelta:+.4f}   p = {p_mdelta:.4f}   "
          f"{'*' if p_mdelta < 0.05 else 'ns'}   n={len(macro_delta_arr)}")
    print(f"    Mean |delta| per sector per week = {mean_delta:.6f} ann. return units "
          f"({mean_delta / MAX_RETURN * 100:.1f}% of max_return)")

    print(f"\n  Mean next-week FULL portfolio return by macro regime label:")
    regime_label_map = {1.0: "risk_on ", 0.0: "neutral ", -1.0: "risk_off"}
    for val in [1.0, 0.0, -1.0]:
        mask = macro_full["macro_regime"] == val
        group = macro_full[mask]["full_return"]
        lbl = regime_label_map[val]
        if len(group) > 0:
            print(f"    {lbl} (scale={INTERCEPT + SLOPE*val:.2f}):  "
                  f"n={len(group):>2}  mean={group.mean()*100:+.3f}%  "
                  f"std={group.std()*100:.3f}%")

    # ================================================================
    # PART 4 — Worst 5 weeks
    # ================================================================
    print("\n" + "=" * 72)
    print("PART 4 — 5 Worst Weeks by LLM Alpha (Full − No-LLM weekly return)")
    print("=" * 72)

    alpha_df = full_weekly.merge(
        nollm_weekly.rename(columns={"weekly_return": "nollm_return"}),
        on="date",
        how="inner",
    )
    alpha_df["alpha"] = alpha_df["weekly_return"] - alpha_df["nollm_return"]
    worst5 = alpha_df.nsmallest(5, "alpha")

    for rank, (_, row) in enumerate(worst5.iterrows(), 1):
        d = row["date"]
        print(f"\n  #{rank}  {d}  alpha={row['alpha']*100:+.2f}%  "
              f"(Full={row['weekly_return']*100:+.2f}%  No-LLM={row['nollm_return']*100:+.2f}%)")

        # News signals this week
        week_news = news_df[news_df["date"] == d].set_index("sector")
        if not week_news.empty:
            avg_s = week_news["sentiment"].mean()
            top_bull = week_news["sentiment"].nlargest(2)
            top_bear = week_news["sentiment"].nsmallest(2)
            print(f"     News avg_sentiment={avg_s:+.3f}")
            print(f"       most bullish: {top_bull.index[0]}={top_bull.iloc[0]:+.2f}  "
                  f"{top_bull.index[1]}={top_bull.iloc[1]:+.2f}")
            print(f"       most bearish: {top_bear.index[0]}={top_bear.iloc[0]:+.2f}  "
                  f"{top_bear.index[1]}={top_bear.iloc[1]:+.2f}")

        # Macro
        regime = macro_dict.get(d, {}).get("macro_regime", None)
        rate = macro_dict.get(d, {}).get("rate_outlook", None)
        regime_lbl = {1.0: "risk_on", 0.0: "neutral", -1.0: "risk_off"}.get(regime, str(regime))
        rate_lbl = {1.0: "rising", 0.0: "stable", -1.0: "falling"}.get(rate, str(rate))
        scale = INTERCEPT + SLOPE * regime if regime is not None else float("nan")
        print(f"     Macro: regime={regime_lbl}  rate_outlook={rate_lbl}  "
              f"→ regime_scale={scale:.2f}")

        # Actual sector returns
        if d in sec_ret_map:
            sec_rets = sec_ret_map[d]
            sorted_secs = sorted(sec_rets.items(), key=lambda x: x[1])
            worst_secs = sorted_secs[:3]
            best_secs = sorted_secs[-3:][::-1]
            print(f"     Market worst:  {', '.join(f'{s}={r*100:+.1f}%' for s,r in worst_secs)}")
            print(f"     Market best:   {', '.join(f'{s}={r*100:+.1f}%' for s,r in best_secs)}")

            # Were agents' bullish/bearish calls directionally right or wrong?
            if not week_news.empty:
                bullish = [s for s in SECTORS if s in week_news.index and week_news.at[s, "sentiment"] > 0.1]
                bearish = [s for s in SECTORS if s in week_news.index and week_news.at[s, "sentiment"] < -0.1]
                bull_correct = sum(1 for s in bullish if sec_rets.get(s, 0) > 0)
                bear_correct = sum(1 for s in bearish if sec_rets.get(s, 0) < 0)
                print(f"     Directional accuracy this week:  "
                      f"bullish {bull_correct}/{len(bullish)} correct  "
                      f"bearish {bear_correct}/{len(bearish)} correct")

    # Weekly alpha chart
    alpha_all = alpha_df.copy()
    worst_dates = set(worst5["date"].tolist())
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(alpha_all)), alpha_all["alpha"] * 100,
           color=["#EF5350" if d in worst_dates else "#2196F3"
                  for d in alpha_all["date"]],
           alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Week (chronological)")
    ax.set_ylabel("LLM alpha (%)")
    ax.set_title("Weekly LLM Alpha: Full vs No-LLM (red = worst 5)")
    tick_step = max(1, len(alpha_all) // 8)
    ax.set_xticks(range(0, len(alpha_all), tick_step))
    ax.set_xticklabels(
        [str(alpha_all.iloc[i]["date"])[:7] for i in range(0, len(alpha_all), tick_step)],
        rotation=30, ha="right", fontsize=7,
    )
    _save_fig(fig, "04_weekly_alpha.png")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  Metric                                  Value")
    print(f"  " + "-" * 55)
    print(f"  News sentiment → sector return (r)      {r_news_pool:+.4f}  p={p_news_pool:.4f}  {'*' if p_news_pool < 0.05 else 'ns'}")
    print(f"  Macro regime → portfolio return (r)     {r_regime:+.4f}  p={p_regime:.4f}  {'*' if p_regime < 0.05 else 'ns'}")
    print(f"  News hit rate (sign match)              {hit_rate:.1%}  p={binom_p:.4f}  {'*' if binom_p < 0.05 else 'ns'}")
    print(f"  Macro regime hit rate                   {mac_rate:.1%}  p={mac_binom_p:.4f}  {'*' if mac_binom_p < 0.05 else 'ns'}")
    print(f"  Q_news_neutral → sector return (r)      {r_qneutral:+.4f}  p={p_qneutral:.4f}  {'*' if p_qneutral < 0.05 else 'ns'}")
    print(f"  Q_actual → sector return (r)            {r_qactual:+.4f}  p={p_qactual:.4f}  {'*' if p_qactual < 0.05 else 'ns'}")
    print(f"  Macro delta → sector return (r)         {r_mdelta:+.4f}  p={p_mdelta:.4f}  {'*' if p_mdelta < 0.05 else 'ns'}")
    print(f"\n  n (pooled, Parts 1–3): {n_obs} (10 sectors × 52 predictive weeks)")
    print(f"  Significance threshold at n={n_obs}: |r| > {1.96/math.sqrt(n_obs):.3f} (p<0.05 two-sided)")
    print(f"\n  Plots saved to: {DIAG_DIR}/")
    print()


if __name__ == "__main__":
    main()
