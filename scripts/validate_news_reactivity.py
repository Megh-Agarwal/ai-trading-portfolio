"""Validate whether NewsAgent reacts to article content or uses model memory.

Two tests per historical date:
  A — Ablation: compare sentiment with articles vs. empty articles.
      If |real - no_news| < 0.15 the agent may be using memory, not content.
  B — Counterfactual: inject 3 fake strongly-contrary headlines for the sector
      with the highest |real sentiment|. If sentiment fails to flip, the model
      ignores article content.

Usage:
    uv run python scripts/validate_news_reactivity.py

Output:
    Prints a table to stdout.
    Saves results to data/validation/news_reactivity.json.
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import anthropic
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from agents.news_agent import NewsAgent
from db.init import init_db
from db.models import NewsRaw

logging.basicConfig(level=logging.WARNING)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"
_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "validation" / "news_reactivity.json"

_COUNTERFACTUAL_NEGATIVE = [
    "Catastrophic earnings miss shocks investors as revenue collapses 40% — CEO resigns",
    "Major regulatory crackdown forces sector-wide production shutdown indefinitely",
    "Credit rating downgrade triggers institutional exodus; sector ETF halted mid-session",
]

_COUNTERFACTUAL_POSITIVE = [
    "Record-breaking earnings beat with revenue surging 80% year-over-year; guidance raised",
    "Landmark government investment package supercharges sector growth for next decade",
    "Regulatory green light for major expansion; analysts upgrade entire sector to overweight",
]


def _get_dates_with_news(db, n: int = 3) -> list[datetime.date]:
    """Return up to n distinct dates from news_raw with the most articles."""
    with Session(db) as s:
        rows = (
            s.execute(
                select(NewsRaw.timestamp)
                .group_by(func.date(NewsRaw.timestamp))
                .order_by(func.count().desc())
                .limit(n)
            )
            .scalars()
            .all()
        )
    return sorted({r.date() if hasattr(r, "date") else r for r in rows})[:n]


def _call_with_input(agent: NewsAgent, input_data: dict, db) -> dict:
    """Invoke the news agent LLM with a custom input dict, bypassing prepare_input."""
    def _call_fn() -> dict:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=agent._model,
            max_tokens=agent._max_tokens,
            temperature=agent._temperature,
            system=agent._prompt,
            tools=[agent._tool],
            tool_choice={"type": "tool", "name": agent._tool["name"]},
            messages=[{"role": "user", "content": json.dumps(input_data, default=str)}],
        )
        return message.model_dump()

    response = agent._cached_call(
        agent._model,
        agent._prompt,
        input_data,
        _call_fn,
        agent_name=agent.agent_name,
        engine=db,
        tool=agent._tool,
    )
    return agent.validate_output(response)


def _build_empty_input(normal_input: dict) -> dict:
    """Return input dict with all articles replaced by empty lists."""
    return {
        **normal_input,
        "sectors": {sector: [] for sector in normal_input["sectors"]},
    }


def _build_counterfactual_input(
    normal_input: dict,
    target_sector: str,
    real_sentiment: float,
    date: datetime.date,
) -> dict:
    """Return input with fake strongly-contrary articles for target_sector."""
    headlines = _COUNTERFACTUAL_NEGATIVE if real_sentiment > 0 else _COUNTERFACTUAL_POSITIVE
    fake_articles = [
        {"timestamp": date.isoformat(), "ticker": "FAKE", "title": h}
        for h in headlines
    ]
    new_sectors = {**normal_input["sectors"], target_sector: fake_articles}
    return {**normal_input, "sectors": new_sectors}


def _is_reactive(real: float, no_news: float, counterfactual: float) -> bool:
    """True if the agent clearly responds to article content."""
    ablation_diff = abs(real - no_news)
    cf_flipped = (real > 0.1 and counterfactual < -0.05) or (real < -0.1 and counterfactual > 0.05)
    return ablation_diff >= 0.15 or cf_flipped


def run_validation(db) -> list[dict]:
    agent = NewsAgent()
    results: list[dict] = []

    dates = _get_dates_with_news(db, n=3)
    if not dates:
        print("No dates with news found in news_raw. Run ingest_news.py first.", file=sys.stderr)
        return []

    for date in dates:
        print(f"\nRunning date={date} ...", flush=True)

        normal_input = agent.prepare_input(date, db)
        total_articles = sum(len(v) for v in normal_input["sectors"].values())
        if total_articles == 0:
            print(f"  Skipping {date}: no articles in window", flush=True)
            continue

        # Normal run
        normal_result = _call_with_input(agent, normal_input, db)

        # Find sector with highest absolute sentiment
        sentiments = normal_result["sector_sentiments"]
        top_sector = max(sentiments, key=lambda s: abs(sentiments[s]))
        real_sentiment = sentiments[top_sector]

        if abs(real_sentiment) < 0.05:
            print(f"  Skipping {date}: all sentiments near zero — no strong signal to test", flush=True)
            continue

        # Ablation run (no articles)
        ablation_input = _build_empty_input(normal_input)
        ablation_result = _call_with_input(agent, ablation_input, db)
        no_news_sentiment = ablation_result["sector_sentiments"][top_sector]

        # Counterfactual run (fake contrary articles for top_sector)
        cf_input = _build_counterfactual_input(normal_input, top_sector, real_sentiment, date)
        cf_result = _call_with_input(agent, cf_input, db)
        cf_sentiment = cf_result["sector_sentiments"][top_sector]

        reactive = _is_reactive(real_sentiment, no_news_sentiment, cf_sentiment)

        row = {
            "date": date.isoformat(),
            "sector": top_sector,
            "real_sentiment": round(real_sentiment, 2),
            "no_news_sentiment": round(no_news_sentiment, 2),
            "counterfactual_sentiment": round(cf_sentiment, 2),
            "reacts_to_news": reactive,
        }
        results.append(row)

    return results


def _print_table(results: list[dict]) -> None:
    if not results:
        print("No results to display.")
        return

    header = f"{'Date':<12} | {'Sector':<6} | {'Real':>8} | {'No-news':>9} | {'Counterfact':>12} | Reacts?"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        sign = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
        yn = "YES" if r["reacts_to_news"] else "NO ⚠"
        print(
            f"{r['date']:<12} | {r['sector']:<6} | {sign(r['real_sentiment']):>8} | "
            f"{sign(r['no_news_sentiment']):>9} | {sign(r['counterfactual_sentiment']):>12} | {yn}"
        )
    print(sep)


def main() -> None:
    if not _DB_PATH.exists():
        print(f"Database not found: {_DB_PATH}", file=sys.stderr)
        sys.exit(1)

    db = create_engine(f"sqlite:///{_DB_PATH}")
    init_db(str(_DB_PATH))

    results = run_validation(db)

    _print_table(results)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
