from __future__ import annotations

import datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Price(Base):
    __tablename__ = "prices"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    adj_close: Mapped[float] = mapped_column(Float)

    __table_args__ = (Index("ix_prices_date_ticker", "date", "ticker"),)


class Macro(Base):
    __tablename__ = "macro"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    series_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[float] = mapped_column(Float)

    __table_args__ = (Index("ix_macro_date_series_id", "date", "series_id"),)


class NewsRaw(Base):
    __tablename__ = "news_raw"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(20))
    sector: Mapped[str | None] = mapped_column(String(50))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime)
    source: Mapped[str | None] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)


class PolymarketRaw(Base):
    __tablename__ = "polymarket_raw"

    market_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    implied_prob: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    category: Mapped[str | None] = mapped_column(String(100))
    end_date: Mapped[datetime.date | None] = mapped_column(Date)


class AgentCall(Base):
    __tablename__ = "agent_calls"

    call_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime)
    agent_name: Mapped[str] = mapped_column(String(100))
    model_string: Mapped[str] = mapped_column(String(100))
    prompt_hash: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    response_json: Mapped[str] = mapped_column(Text)
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime.date] = mapped_column(Date)
    agent_name: Mapped[str] = mapped_column(String(100))
    target: Mapped[str] = mapped_column(String(50))
    signal_value: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    raw_call_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("agent_calls.call_id"))


class View(Base):
    __tablename__ = "views"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    sector: Mapped[str] = mapped_column(String(50), primary_key=True)
    expected_return: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)


class TargetWeight(Base):
    __tablename__ = "target_weights"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    sector: Mapped[str] = mapped_column(String(50), primary_key=True)
    weight: Mapped[float] = mapped_column(Float)


class Trade(Base):
    __tablename__ = "trades"

    trade_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime.date] = mapped_column(Date)
    ticker: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))  # "buy" | "sell"
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    slippage: Mapped[float] = mapped_column(Float)


class Position(Base):
    __tablename__ = "positions"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    shares: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    cost_basis: Mapped[float] = mapped_column(Float)

    __table_args__ = (Index("ix_positions_date_ticker", "date", "ticker"),)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshot"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    total_value: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    gross_exposure: Mapped[float] = mapped_column(Float)
    net_exposure: Mapped[float] = mapped_column(Float)

    __table_args__ = (Index("ix_portfolio_snapshot_date", "date"),)
