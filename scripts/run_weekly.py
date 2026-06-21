"""Weekly rebalance CLI entry point — Ticket 4.5.

Usage:
  uv run python scripts/run_weekly.py
  uv run python scripts/run_weekly.py --date 2024-06-07 --mode backtest
  uv run python scripts/run_weekly.py --date 2024-06-07 --force

Exit codes: 0 = success or idempotent skip, non-zero = unhandled failure.
Full traceback on failure goes to stdout so CloudWatch can capture it (M6).
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import create_engine  # noqa: E402

from db.init import init_db  # noqa: E402
from weekly_run import run_weekly  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "state.db"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the weekly portfolio rebalance.")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Rebalance date (default: today).",
    )
    p.add_argument(
        "--mode",
        choices=["backtest", "live"],
        default="backtest",
        help="backtest = no Polymarket; live = full signal set (default: backtest).",
    )
    p.add_argument(
        "--db-path",
        default=str(_DB_PATH),
        metavar="PATH",
        help="Path to SQLite state.db (default: data/state.db).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if this date was already rebalanced.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    date_str = args.date or datetime.date.today().isoformat()

    db_path = Path(args.db_path)
    init_db(db_path)
    db_engine = create_engine(f"sqlite:///{db_path}")

    try:
        result = run_weekly(
            date_str=date_str,
            mode=args.mode,
            db_engine=db_engine,
            force=args.force,
        )
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
    except Exception:
        # Full traceback to stdout so CloudWatch captures it alongside normal logs
        print("CRITICAL: weekly rebalance failed", file=sys.stderr)
        print(traceback.format_exc())
        sys.exit(1)
    finally:
        db_engine.dispose()


if __name__ == "__main__":
    main()
