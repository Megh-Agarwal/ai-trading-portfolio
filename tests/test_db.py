from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from db.init import init_db
from db.models import AgentCall, Base, Price, Signal

EXPECTED_TABLES = {
    "prices",
    "macro",
    "news_raw",
    "polymarket_raw",
    "agent_calls",
    "signals",
    "views",
    "target_weights",
    "trades",
    "positions",
    "portfolio_snapshot",
    "risk_events",
}


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def test_all_tables_created(engine):
    tables = set(inspect(engine).get_table_names())
    assert tables == EXPECTED_TABLES


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "state.db"
    init_db(db_path)
    init_db(db_path)  # must not raise


def test_init_db_creates_file(tmp_path):
    db_path = tmp_path / "state.db"
    assert not db_path.exists()
    init_db(db_path)
    assert db_path.exists()


def test_price_insert_and_query(engine):
    row = Price(
        date=datetime.date(2024, 1, 2),
        ticker="XLK",
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=1_000_000,
        adj_close=103.0,
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()

    with Session(engine) as session:
        fetched = session.get(Price, (datetime.date(2024, 1, 2), "XLK"))
        assert fetched is not None
        assert fetched.close == 103.0
        assert fetched.volume == 1_000_000


def test_agent_call_and_signal_fk(engine):
    call = AgentCall(
        timestamp=datetime.datetime(2024, 1, 2, 12, 0),
        agent_name="sentiment",
        model_string="claude-haiku-4-5",
        prompt_hash="a" * 64,
        input_hash="b" * 64,
        response_json='{"score": 0.8}',
        tokens_in=100,
        tokens_out=20,
    )
    with Session(engine) as session:
        session.add(call)
        session.flush()
        signal = Signal(
            date=datetime.date(2024, 1, 2),
            agent_name="sentiment",
            target="XLK",
            signal_value=0.8,
            confidence=0.9,
            raw_call_id=call.call_id,
        )
        session.add(signal)
        session.commit()
        call_id = call.call_id

    with Session(engine) as session:
        sig = session.query(Signal).filter_by(raw_call_id=call_id).one()
        assert sig.signal_value == 0.8
