"""Tests for src/api/main.py — Ticket 6.5."""

from __future__ import annotations

import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from api.main import app, get_engine
from db.models import (
    PORTFOLIO_LIVE,
    Base,
    PortfolioSnapshot,
    Position,
    RiskEvent,
    Signal,
    Trade,
)

_TODAY = datetime.date(2026, 6, 28)
_LAST_WEEK = datetime.date(2026, 6, 21)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine() -> Engine:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # Two snapshots so /performance can compute returns
        session.add_all([
            PortfolioSnapshot(
                portfolio_id=PORTFOLIO_LIVE,
                date=_LAST_WEEK,
                total_value=1_000_000.0,
                cash=300.0,
                gross_exposure=0.999,
                net_exposure=0.999,
            ),
            PortfolioSnapshot(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                total_value=999_700.0,
                cash=246.72,
                gross_exposure=0.9997,
                net_exposure=0.9997,
            ),
        ])
        # Positions for latest date
        session.add_all([
            Position(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                ticker="XLK",
                shares=120.0,
                market_value=250_000.0,
                cost_basis=249_700.0,
            ),
            Position(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                ticker="CASH",
                shares=246.72,
                market_value=246.72,
                cost_basis=246.72,
            ),
        ])
        # Signals for latest date
        session.add_all([
            Signal(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                agent_name="sentiment",
                target="XLK",
                signal_value=0.8,
                confidence=0.7,
            ),
            Signal(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                agent_name="macro",
                target="regime",
                signal_value=0.0,
                confidence=0.62,
            ),
            Signal(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                agent_name="macro",
                target="rate_outlook",
                signal_value=0.0,
                confidence=0.62,
            ),
            Signal(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                agent_name="events",
                target="XLK",
                signal_value=0.3,
                confidence=0.65,
            ),
        ])
        # Trades
        session.add_all([
            Trade(
                portfolio_id=PORTFOLIO_LIVE,
                date=_TODAY,
                ticker="XLK",
                side="buy",
                shares=120.0,
                price=2083.33,
                commission=62.5,
                slippage=0.0,
            ),
        ])
        # Risk event (not triggered)
        session.add(RiskEvent(
            portfolio_id=PORTFOLIO_LIVE,
            date=_TODAY,
            check_name="max_position",
            triggered=False,
            value=0.25,
            threshold=0.25,
            action_taken=None,
            message=None,
        ))
        session.commit()
    return engine


@pytest.fixture()
def client(db_engine: Engine) -> TestClient:
    app.dependency_overrides[get_engine] = lambda: db_engine
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /portfolio
# ---------------------------------------------------------------------------


def test_portfolio_returns_nav(client: TestClient) -> None:
    r = client.get("/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["nav"] == 999_700.0
    assert body["as_of_date"] == "2026-06-28"
    assert body["cash"] == 246.72


def test_portfolio_positions_exclude_cash(client: TestClient) -> None:
    r = client.get("/portfolio")
    tickers = [p["ticker"] for p in r.json()["positions"]]
    assert "CASH" not in tickers
    assert "XLK" in tickers


def test_portfolio_weights_sum_to_one_approx(client: TestClient) -> None:
    r = client.get("/portfolio")
    positions = r.json()["positions"]
    total_weight = sum(p["weight"] for p in positions)
    # XLK is ~25% of NAV; cash is excluded; so weight < 1 is expected
    assert 0.0 < total_weight <= 1.0


def test_portfolio_empty_db_returns_404() -> None:
    empty_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(empty_engine)
    app.dependency_overrides[get_engine] = lambda: empty_engine
    try:
        r = TestClient(app).get("/portfolio")
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /performance
# ---------------------------------------------------------------------------


def test_performance_returns_nav_history(client: TestClient) -> None:
    r = client.get("/performance")
    assert r.status_code == 200
    body = r.json()
    assert body["weeks"] == 2
    dates = [row["date"] for row in body["nav_history"]]
    assert "2026-06-21" in dates
    assert "2026-06-28" in dates


def test_performance_includes_metrics(client: TestClient) -> None:
    r = client.get("/performance")
    body = r.json()
    assert "metrics" in body
    assert "total_return" in body["metrics"]


def test_performance_empty_db_returns_404() -> None:
    empty_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(empty_engine)
    app.dependency_overrides[get_engine] = lambda: empty_engine
    try:
        r = TestClient(app).get("/performance")
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /signals/latest
# ---------------------------------------------------------------------------


def test_signals_latest_returns_all_agents(client: TestClient) -> None:
    r = client.get("/signals/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["date"] == "2026-06-28"
    assert "XLK" in body["sentiment"]
    assert "regime" in body["macro"]
    assert "XLK" in body["events"]


def test_signals_values_are_floats(client: TestClient) -> None:
    r = client.get("/signals/latest")
    body = r.json()
    assert isinstance(body["sentiment"]["XLK"], float)
    assert isinstance(body["events"]["XLK"], float)


def test_signals_empty_db_returns_404() -> None:
    empty_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(empty_engine)
    app.dependency_overrides[get_engine] = lambda: empty_engine
    try:
        r = TestClient(app).get("/signals/latest")
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /trades/recent
# ---------------------------------------------------------------------------


def test_trades_recent_default_limit(client: TestClient) -> None:
    r = client.get("/trades/recent")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    assert body["trades"][0]["ticker"] == "XLK"
    assert body["trades"][0]["side"] == "buy"


def test_trades_recent_n_param(client: TestClient) -> None:
    r = client.get("/trades/recent?n=5")
    assert r.status_code == 200
    assert r.json()["n"] <= 5


def test_trades_recent_n_out_of_range(client: TestClient) -> None:
    r = client.get("/trades/recent?n=0")
    assert r.status_code == 422  # FastAPI validation error

    r = client.get("/trades/recent?n=101")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_healthy_when_recent_run(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["last_run_date"] == "2026-06-28"
    assert isinstance(body["days_since_last_run"], int)


def test_health_stale_when_old_run() -> None:
    stale_engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(stale_engine)
    old_date = datetime.date(2020, 1, 1)
    with Session(stale_engine) as session:
        session.add(PortfolioSnapshot(
            portfolio_id=PORTFOLIO_LIVE,
            date=old_date,
            total_value=1_000_000.0,
            cash=0.0,
            gross_exposure=1.0,
            net_exposure=1.0,
        ))
        session.commit()

    app.dependency_overrides[get_engine] = lambda: stale_engine
    try:
        r = TestClient(app).get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "stale"
    finally:
        app.dependency_overrides.clear()


def test_health_never_run_when_empty_db() -> None:
    empty_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(empty_engine)
    app.dependency_overrides[get_engine] = lambda: empty_engine
    try:
        r = TestClient(app).get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "never_run"
        assert body["last_run_date"] is None
    finally:
        app.dependency_overrides.clear()


def test_health_no_triggered_risk_events(client: TestClient) -> None:
    r = client.get("/health")
    # The fixture adds a non-triggered risk event — should not appear
    assert r.json()["triggered_risk_events"] == []
