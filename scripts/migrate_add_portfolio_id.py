"""Migration: add portfolio_id to all rebalance tables (Ticket 5.1).

Idempotent — safe to re-run; skips each table if portfolio_id already present.
Existing rows are tagged portfolio_id='live' to preserve the 2026-06-13 manual
test data and keep it cleanly separated from backtest runs.

Tables with autoincrement PKs (signals, trades, risk_events):
  ALTER TABLE ... ADD COLUMN portfolio_id TEXT NOT NULL DEFAULT 'live'

Tables with composite PKs that include portfolio_id
(views, target_weights, positions, portfolio_snapshot):
  Recreated via rename → create → copy → drop (SQLite cannot alter PKs).

Usage:
  uv run python scripts/migrate_add_portfolio_id.py [--db-path PATH]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "state.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if `column` exists in `table`."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _add_column_if_missing(conn: sqlite3.Connection, table: str) -> bool:
    """Add portfolio_id TEXT NOT NULL DEFAULT 'live' if absent. Returns True if added."""
    if _has_column(conn, table, "portfolio_id"):
        print(f"  {table}: portfolio_id already present — skip")
        return False
    conn.execute(
        f"ALTER TABLE {table} ADD COLUMN portfolio_id TEXT NOT NULL DEFAULT 'live'"
    )
    print(f"  {table}: added portfolio_id column")
    return True


def _recreate_table_with_portfolio_id(
    conn: sqlite3.Connection,
    table: str,
    create_sql: str,
    copy_sql: str,
    index_sqls: list[str],
) -> None:
    """Recreate `table` with portfolio_id as part of PK.

    Steps: rename old → create new → copy data → drop old → create indices.
    Skips entirely if portfolio_id already present.
    """
    if _has_column(conn, table, "portfolio_id"):
        print(f"  {table}: portfolio_id already present — skip")
        return
    old = f"{table}_pre_migration"
    conn.execute(f"ALTER TABLE {table} RENAME TO {old}")
    conn.execute(create_sql)
    conn.execute(copy_sql.format(old=old))
    conn.execute(f"DROP TABLE {old}")
    for idx_sql in index_sqls:
        conn.execute(idx_sql)
    print(f"  {table}: recreated with portfolio_id in PK, existing rows → portfolio_id='live'")


# ---------------------------------------------------------------------------
# Per-table migration definitions
# ---------------------------------------------------------------------------


def migrate_views(conn: sqlite3.Connection) -> None:
    _recreate_table_with_portfolio_id(
        conn,
        "views",
        """
        CREATE TABLE views (
            portfolio_id TEXT NOT NULL DEFAULT 'live',
            date         DATE NOT NULL,
            sector       TEXT NOT NULL,
            expected_return REAL NOT NULL,
            confidence   REAL,
            PRIMARY KEY (portfolio_id, date, sector)
        )
        """,
        "INSERT INTO views SELECT 'live', date, sector, expected_return, confidence FROM {old}",
        [],
    )


def migrate_target_weights(conn: sqlite3.Connection) -> None:
    _recreate_table_with_portfolio_id(
        conn,
        "target_weights",
        """
        CREATE TABLE target_weights (
            portfolio_id TEXT NOT NULL DEFAULT 'live',
            date         DATE NOT NULL,
            sector       TEXT NOT NULL,
            weight       REAL NOT NULL,
            PRIMARY KEY (portfolio_id, date, sector)
        )
        """,
        "INSERT INTO target_weights SELECT 'live', date, sector, weight FROM {old}",
        [],
    )


def migrate_positions(conn: sqlite3.Connection) -> None:
    _recreate_table_with_portfolio_id(
        conn,
        "positions",
        """
        CREATE TABLE positions (
            portfolio_id TEXT NOT NULL DEFAULT 'live',
            date         DATE NOT NULL,
            ticker       TEXT NOT NULL,
            shares       REAL NOT NULL,
            market_value REAL NOT NULL,
            cost_basis   REAL NOT NULL,
            PRIMARY KEY (portfolio_id, date, ticker)
        )
        """,
        "INSERT INTO positions SELECT 'live', date, ticker, shares, market_value, cost_basis FROM {old}",
        ["CREATE INDEX IF NOT EXISTS ix_positions_portfolio_date_ticker ON positions (portfolio_id, date, ticker)"],
    )


def migrate_portfolio_snapshot(conn: sqlite3.Connection) -> None:
    _recreate_table_with_portfolio_id(
        conn,
        "portfolio_snapshot",
        """
        CREATE TABLE portfolio_snapshot (
            portfolio_id  TEXT NOT NULL DEFAULT 'live',
            date          DATE NOT NULL,
            total_value   REAL NOT NULL,
            cash          REAL NOT NULL,
            gross_exposure REAL NOT NULL,
            net_exposure  REAL NOT NULL,
            PRIMARY KEY (portfolio_id, date)
        )
        """,
        "INSERT INTO portfolio_snapshot SELECT 'live', date, total_value, cash, gross_exposure, net_exposure FROM {old}",
        ["CREATE INDEX IF NOT EXISTS ix_portfolio_snapshot_portfolio_date ON portfolio_snapshot (portfolio_id, date)"],
    )


def migrate_signals(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "signals")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_signals_portfolio_date ON signals (portfolio_id, date)"
    )


def migrate_trades(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "trades")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_trades_portfolio_date ON trades (portfolio_id, date)"
    )


def migrate_risk_events(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "risk_events")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_risk_events_portfolio_date ON risk_events (portfolio_id, date)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_migration(db_path: Path) -> None:
    if not db_path.exists():
        print(f"DB not found at {db_path} — nothing to migrate.")
        return

    print(f"Migrating {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        # Recreations must run inside a transaction that can be rolled back on failure
        conn.execute("BEGIN")
        migrate_signals(conn)
        migrate_trades(conn)
        migrate_risk_events(conn)
        migrate_views(conn)
        migrate_target_weights(conn)
        migrate_positions(conn)
        migrate_portfolio_snapshot(conn)
        conn.execute("COMMIT")
        print("Migration complete.")
    except Exception as exc:
        conn.execute("ROLLBACK")
        print(f"Migration FAILED — rolled back. Error: {exc}", file=sys.stderr)
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=str(_DEFAULT_DB),
        help=f"Path to state.db (default: {_DEFAULT_DB})",
    )
    args = parser.parse_args()
    run_migration(Path(args.db_path))


if __name__ == "__main__":
    main()
