"""Neon FinBrain single-file edition.

Install:
    pip install nicegui fastapi httpx numpy pandas plotly scikit-learn yfinance feedparser pydantic

Run:
    python neon_finbrain.py

Open:
    http://localhost:8080
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import time
from urllib.parse import quote, urlparse

import feedparser
import httpx
from fastapi.encoders import jsonable_encoder
from nicegui import app, ui
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yfinance as yf


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")


def split_csv(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    app_name: str = "Neon FinBrain"
    refresh_seconds: float = float(os.getenv("REFRESH_SECONDS", "3"))
    market_interval: str = os.getenv("MARKET_INTERVAL", "1m")
    market_period: str = os.getenv("MARKET_PERIOD", "5d")
    max_chart_points: int = int(os.getenv("MAX_CHART_POINTS", "240"))
    watchlist: list[str] = field(
        default_factory=lambda: split_csv(
            os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,TSLA,SPY,QQQ,AMD,AMZN")
        )
    )
    max_news_items: int = int(os.getenv("MAX_NEWS_ITEMS", "6"))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
    )
    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DB_PATH", str(BASE_DIR / "data" / "financial_brain.db"))
        ).resolve()
    )
    polygon_api_key: str | None = os.getenv("POLYGON_API_KEY")
    alpaca_api_key: str | None = os.getenv("ALPACA_API_KEY")
    alpaca_api_secret: str | None = os.getenv("ALPACA_API_SECRET")
    robinhood_username: str | None = os.getenv("ROBINHOOD_USERNAME")
    robinhood_password: str | None = os.getenv("ROBINHOOD_PASSWORD")
    plaid_client_id: str | None = os.getenv("PLAID_CLIENT_ID")
    plaid_secret: str | None = os.getenv("PLAID_SECRET")
    plaid_access_token: str | None = os.getenv("PLAID_ACCESS_TOKEN")
    plaid_base_url: str = os.getenv("PLAID_BASE_URL", "https://sandbox.plaid.com")
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "demo-user")
    trusted_news_domains: tuple[str, ...] = (
        "finance.yahoo.com",
        "reuters.com",
        "wsj.com",
        "bloomberg.com",
        "marketwatch.com",
        "cnbc.com",
        "seekingalpha.com",
        "investing.com",
        "fool.com",
        "benzinga.com",
        "thestreet.com",
    )

    def ensure_paths(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_paths()


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalIndicators(BaseModel):
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    atr: float = 0.0
    volume_spike: float = 1.0
    trend: str = "neutral"
    patterns: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    label: str = "neutral"
    horizon: str = "next-3-bars"
    probability_up: float = 0.5
    confidence: float = 0.5
    expected_move_pct: float = 0.0
    model_version: str = "heuristic"
    drivers: list[str] = Field(default_factory=list)


class QuoteSnapshot(BaseModel):
    ticker: str
    source: str
    last: float
    previous_close: float
    change: float
    change_percent: float
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    volume: float = 0.0
    avg_volume: float = 0.0
    volatility: float = 0.0
    last_updated: datetime
    candles: list[Candle] = Field(default_factory=list)
    technicals: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    prediction: Prediction = Field(default_factory=Prediction)


class NewsItem(BaseModel):
    ticker: str
    title: str
    url: str
    source: str
    published_at: datetime
    summary: str = ""
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    relevance: float = 0.5
    credibility: float = 0.5
    flags: list[str] = Field(default_factory=list)


class PortfolioPosition(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    market_price: float
    market_value: float
    pnl_unrealized: float
    allocation: float
    risk_score: float
    thesis: str = ""


class ExpenseItem(BaseModel):
    category: str
    amount: float
    merchant: str
    incurred_on: date


class PortfolioSummary(BaseModel):
    total_value: float = 0.0
    cash: float = 0.0
    day_pnl: float = 0.0
    total_pnl: float = 0.0
    risk_score: float = 0.0
    diversification_score: float = 0.0
    expenses_month: float = 0.0
    positions: list[PortfolioPosition] = Field(default_factory=list)
    expenses: list[ExpenseItem] = Field(default_factory=list)
    connector_status: dict[str, str] = Field(default_factory=dict)
    budget_advice: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    user_id: str
    name: str = "Operator"
    investment_horizon: str = "balanced"
    risk_tolerance: str = "moderate"
    liquidity_need: str = "medium"
    monthly_budget: float = 2500.0
    cash_balance: float = 10000.0
    watchlist: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    sectors_of_interest: list[str] = Field(default_factory=list)
    avoid_tickers: list[str] = Field(default_factory=list)


class Insight(BaseModel):
    level: str
    ticker: str
    headline: str
    rationale: str
    risk_level: str
    confidence: float
    action: str
    horizon: str
    drivers: list[str] = Field(default_factory=list)
    timestamp: datetime


class BrainDecision(BaseModel):
    market_regime: str = "neutral"
    summary: str = ""
    risk_posture: str = "balanced"
    confidence: float = 0.5
    opportunities: list[Insight] = Field(default_factory=list)
    warnings: list[Insight] = Field(default_factory=list)
    top_day_trade_tickers: list[str] = Field(default_factory=list)
    top_long_term_tickers: list[str] = Field(default_factory=list)
    strategy_notes: list[str] = Field(default_factory=list)
    market_sentiment_score: float = 0.0


class ChatTurn(BaseModel):
    role: str
    message: str
    timestamp: datetime


class ChatReply(BaseModel):
    response: str
    suggestions: list[str] = Field(default_factory=list)
    referenced_tickers: list[str] = Field(default_factory=list)
    updated_profile: UserProfile | None = None


class ScenarioImpact(BaseModel):
    ticker: str
    shock_pct: float
    estimated_pnl: float


class ScenarioResult(BaseModel):
    title: str
    market_shock_pct: float
    estimated_portfolio_change: float
    estimated_change_pct: float
    impacts: list[ScenarioImpact] = Field(default_factory=list)
    narrative: str = ""


class SimulatedTrade(BaseModel):
    ticker: str
    side: str
    shares: float
    price: float
    created_at: datetime


class SimulationPosition(BaseModel):
    ticker: str
    net_shares: float
    avg_entry: float
    market_price: float
    exposure: float
    pnl_unrealized: float


class SimulationBook(BaseModel):
    equity: float = 0.0
    gross_exposure: float = 0.0
    positions: list[SimulationPosition] = Field(default_factory=list)
    trades: list[SimulatedTrade] = Field(default_factory=list)


class PositionRequest(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    thesis: str = ""


class ExpenseRequest(BaseModel):
    category: str
    amount: float
    merchant: str = ""


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    goal_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    thesis TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, ticker)
                );
                CREATE TABLE IF NOT EXISTS expense_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    merchant TEXT NOT NULL,
                    incurred_on TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sim_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def load_user_profile(self, user_id: str, watchlist: list[str]) -> UserProfile:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row:
            return UserProfile.model_validate_json(row["payload"])
        profile = UserProfile(user_id=user_id, watchlist=watchlist)
        self.save_user_profile(profile)
        return profile

    def save_user_profile(self, profile: UserProfile) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles (user_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.model_dump_json(),
                    datetime.utcnow().isoformat(),
                ),
            )

    def add_daily_goal(self, user_id: str, goal_text: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO daily_goals (user_id, goal_text, created_at) VALUES (?, ?, ?)",
                (user_id, goal_text, datetime.utcnow().isoformat()),
            )

    def add_chat_message(self, user_id: str, role: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (user_id, role, message, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, role, message, datetime.utcnow().isoformat()),
            )

    def list_recent_chat_messages(self, user_id: str, limit: int = 15) -> list[ChatTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, message, created_at
                FROM chat_messages
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            ChatTurn(
                role=row["role"],
                message=row["message"],
                timestamp=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
        ]

    def upsert_position(self, user_id: str, ticker: str, shares: float, avg_cost: float, thesis: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO portfolio_positions
                    (user_id, ticker, shares, avg_cost, thesis, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, ticker) DO UPDATE SET
                    shares = excluded.shares,
                    avg_cost = excluded.avg_cost,
                    thesis = excluded.thesis,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    ticker.upper(),
                    shares,
                    avg_cost,
                    thesis,
                    datetime.utcnow().isoformat(),
                ),
            )

    def list_positions(self, user_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ticker, shares, avg_cost, thesis
                FROM portfolio_positions
                WHERE user_id = ?
                ORDER BY ticker
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_expense(self, user_id: str, category: str, amount: float, merchant: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO expense_entries
                    (user_id, category, amount, merchant, incurred_on)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, category, amount, merchant, date.today().isoformat()),
            )

    def list_expenses(self, user_id: str, days: int = 31) -> list[ExpenseItem]:
        cutoff = date.today() - timedelta(days=days)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT category, amount, merchant, incurred_on
                FROM expense_entries
                WHERE user_id = ? AND incurred_on >= ?
                ORDER BY incurred_on DESC
                """,
                (user_id, cutoff.isoformat()),
            ).fetchall()
        return [
            ExpenseItem(
                category=row["category"],
                amount=row["amount"],
                merchant=row["merchant"],
                incurred_on=date.fromisoformat(row["incurred_on"]),
            )
            for row in rows
        ]

    def add_sim_trade(self, user_id: str, ticker: str, side: str, shares: float, price: float) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sim_trades (user_id, ticker, side, shares, price, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    ticker.upper(),
                    side.upper(),
                    shares,
                    price,
                    datetime.utcnow().isoformat(),
                ),
            )

    def list_sim_trades(self, user_id: str, limit: int = 100) -> list[SimulatedTrade]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ticker, side, shares, price, created_at
                FROM sim_trades
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            SimulatedTrade(
                ticker=row["ticker"],
                side=row["side"],
                shares=row["shares"],
                price=row["price"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reversed(rows)
        ]


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.DataFrame([c.model_dump() for c in candles]).set_index("timestamp").sort_index()


def compute_volatility(candles: list[Candle]) -> float:
    frame = candles_to_frame(candles)
    if frame.empty or len(frame) < 5:
        return 0.0
    returns = frame["close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.tail(30).std() * math.sqrt(min(len(frame), 252)))


def compute_technicals(candles: list[Candle]) -> TechnicalIndicators:
    frame = candles_to_frame(candles)
    if frame.empty:
        return TechnicalIndicators()

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"].replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean().replace(0, np.nan)
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    sma_20 = close.rolling(20, min_periods=1).mean()
    sma_50 = close.rolling(50, min_periods=1).mean()

    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=1).mean()

    avg_volume = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
    volume_spike = (volume / avg_volume).fillna(1.0)

    latest_close = float(close.iloc[-1])
    latest_sma20 = float(sma_20.iloc[-1])
    latest_sma50 = float(sma_50.iloc[-1])
    latest_rsi = float(rsi.iloc[-1])
    latest_macd = float(macd.iloc[-1])
    latest_signal = float(macd_signal.iloc[-1])

    trend = "neutral"
    if latest_close > latest_sma20 > latest_sma50 and latest_macd >= latest_signal:
        trend = "bullish"
    elif latest_close < latest_sma20 < latest_sma50 and latest_macd <= latest_signal:
        trend = "bearish"

    patterns: list[str] = []
    if latest_rsi < 35 and latest_close >= latest_sma20 * 0.99:
        patterns.append("oversold-reversal")
    if latest_rsi > 68 and latest_close < latest_sma20:
        patterns.append("momentum-exhaustion")
    if len(close) > 20 and latest_close > float(close.tail(20).iloc[:-1].max()):
        patterns.append("breakout")
    if len(close) > 20 and latest_close < float(close.tail(20).iloc[:-1].min()):
        patterns.append("breakdown")
    if len(sma_20) > 2 and len(sma_50) > 2:
        previous_gap = float(sma_20.iloc[-2] - sma_50.iloc[-2])
        current_gap = float(sma_20.iloc[-1] - sma_50.iloc[-1])
        if previous_gap <= 0 < current_gap:
            patterns.append("golden-cross")
        if previous_gap >= 0 > current_gap:
            patterns.append("death-cross")
    if float(volume_spike.iloc[-1]) >= 1.8:
        patterns.append("volume-spike")

    return TechnicalIndicators(
        rsi=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_signal,
        sma_20=latest_sma20,
        sma_50=latest_sma50,
        ema_12=float(ema_12.iloc[-1]),
        ema_26=float(ema_26.iloc[-1]),
        atr=float(atr.iloc[-1]),
        volume_spike=float(volume_spike.iloc[-1]),
        trend=trend,
        patterns=patterns,
    )


def build_feature_frame(candles: list[Candle]) -> pd.DataFrame:
    frame = candles_to_frame(candles)
    if frame.empty or len(frame) < 40:
        return pd.DataFrame()

    close = frame["close"]
    volume = frame["volume"].replace(0, np.nan).ffill().fillna(0)
    technicals = compute_technicals(candles)

    feature_frame = pd.DataFrame(index=frame.index)
    feature_frame["return_1"] = close.pct_change()
    feature_frame["return_3"] = close.pct_change(3)
    feature_frame["return_10"] = close.pct_change(10)
    feature_frame["volatility_10"] = close.pct_change().rolling(10, min_periods=1).std()
    feature_frame["volume_ratio"] = volume / volume.rolling(20, min_periods=1).mean()
    feature_frame["sma_gap"] = (close / close.rolling(20, min_periods=1).mean()) - 1
    feature_frame["ema_gap"] = (close / close.ewm(span=12, adjust=False).mean()) - 1
    feature_frame["rsi_like"] = (
        close.diff().clip(lower=0).rolling(14, min_periods=1).mean()
        / close.diff().abs().rolling(14, min_periods=1).mean().replace(0, np.nan)
    ).fillna(0)
    feature_frame["atr_ratio"] = technicals.atr / close.replace(0, np.nan)
    feature_frame["future_return"] = close.shift(-3) / close - 1
    return feature_frame.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass(slots=True)
class CachedModel:
    model: Pipeline
    trained_at: float
    sample_size: int


class PredictionEngine:
    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}

    def predict(self, ticker: str, candles: list[Candle]) -> Prediction:
        features = build_feature_frame(candles)
        if len(features) < 50:
            return Prediction(
                label="neutral",
                probability_up=0.5,
                confidence=0.4,
                model_version="fallback",
                drivers=["not-enough-history"],
            )

        model = self._get_or_train_model(ticker, features)
        feature_columns = [col for col in features.columns if col != "future_return"]
        latest = features[feature_columns].tail(1)
        probability_up = float(model.predict_proba(latest)[0][1])
        expected_move_pct = float(features["future_return"].tail(40).std() * (probability_up - 0.5) * 12 * 100)
        label = "neutral"
        if probability_up >= 0.57:
            label = "bullish"
        elif probability_up <= 0.43:
            label = "bearish"
        confidence = min(0.95, 0.45 + abs(probability_up - 0.5) * 1.7)
        drivers: list[str] = []
        importances = getattr(model.named_steps["forest"], "feature_importances_", [])
        if len(importances) == len(feature_columns):
            ranked = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
            drivers = [name for name, _ in ranked[:3]]
        return Prediction(
            label=label,
            probability_up=probability_up,
            confidence=confidence,
            expected_move_pct=expected_move_pct,
            model_version="rf-v1-free",
            drivers=drivers,
        )

    def _get_or_train_model(self, ticker: str, features: pd.DataFrame) -> Pipeline:
        cached = self._cache.get(ticker)
        if cached and time() - cached.trained_at < 1800 and cached.sample_size == len(features):
            return cached.model

        feature_columns = [column for column in features.columns if column != "future_return"]
        target = (features["future_return"] > 0).astype(int)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(features[feature_columns], target)
        self._cache[ticker] = CachedModel(model=model, trained_at=time(), sample_size=len(features))
        return model


def to_optional_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class YahooFinanceProvider:
    name = "yfinance"

    def __init__(self, config: Settings) -> None:
        self.config = config

    async def get_snapshot(self, ticker: str) -> QuoteSnapshot:
        return await asyncio.to_thread(self._get_snapshot_sync, ticker)

    def _get_snapshot_sync(self, ticker: str) -> QuoteSnapshot:
        instrument = yf.Ticker(ticker)
        history = instrument.history(
            period=self.config.market_period,
            interval=self.config.market_interval,
            auto_adjust=False,
            prepost=True,
        )
        if history.empty:
            raise ValueError(f"No data returned for {ticker}")

        history = history.tail(self.config.max_chart_points)
        candles = self._to_candles(history)
        fast_info = dict(getattr(instrument, "fast_info", {}) or {})
        last = float(fast_info.get("lastPrice") or history["Close"].iloc[-1])
        previous_close = to_optional_float(fast_info.get("previousClose"))
        if previous_close is None:
            previous_close = float(history["Close"].iloc[-2]) if len(history) > 1 else last
        bid = to_optional_float(fast_info.get("bid"))
        ask = to_optional_float(fast_info.get("ask"))
        spread = ask - bid if bid is not None and ask is not None else None
        volume = float(fast_info.get("lastVolume") or history["Volume"].iloc[-1] or 0)
        avg_volume = float(history["Volume"].tail(30).mean() or 0)
        change = last - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0.0

        return QuoteSnapshot(
            ticker=ticker,
            source=self.name,
            last=last,
            previous_close=previous_close,
            change=change,
            change_percent=change_percent,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=volume,
            avg_volume=avg_volume,
            volatility=compute_volatility(candles),
            last_updated=datetime.now(UTC),
            candles=candles,
            technicals=compute_technicals(candles),
        )

    @staticmethod
    def _to_candles(frame: pd.DataFrame) -> list[Candle]:
        candles: list[Candle] = []
        for timestamp, row in frame.iterrows():
            ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            candles.append(
                Candle(
                    timestamp=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"] or 0),
                )
            )
        return candles


class PolygonProvider:
    name = "polygon"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def get_snapshot(self, ticker: str) -> QuoteSnapshot:
        end = datetime.now(UTC)
        start = end - timedelta(days=5)
        bars_url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
            f"{start.date()}/{end.date()}?adjusted=true&sort=asc&limit=500&apiKey={self.api_key}"
        )
        trade_url = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={self.api_key}"
        quote_url = f"https://api.polygon.io/v2/last/nbbo/{ticker}?apiKey={self.api_key}"
        async with httpx.AsyncClient(timeout=10) as client:
            bars_resp, trade_resp, quote_resp = await asyncio.gather(
                client.get(bars_url),
                client.get(trade_url),
                client.get(quote_url),
            )
        bars_resp.raise_for_status()
        trade_resp.raise_for_status()
        quote_resp.raise_for_status()
        bars_data = bars_resp.json().get("results", [])
        trade_data = trade_resp.json().get("results", {})
        quote_data = quote_resp.json().get("results", {})
        if not bars_data:
            raise ValueError(f"No polygon bars for {ticker}")

        candles = [
            Candle(
                timestamp=datetime.fromtimestamp(item["t"] / 1000, tz=UTC),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
            )
            for item in bars_data[-240:]
        ]
        last = float(trade_data.get("p") or candles[-1].close)
        previous_close = candles[-2].close if len(candles) > 1 else last
        bid = to_optional_float(quote_data.get("P"))
        ask = to_optional_float(quote_data.get("p"))
        spread = ask - bid if bid is not None and ask is not None else None
        return QuoteSnapshot(
            ticker=ticker,
            source=self.name,
            last=last,
            previous_close=previous_close,
            change=last - previous_close,
            change_percent=((last - previous_close) / previous_close * 100) if previous_close else 0.0,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=candles[-1].volume,
            avg_volume=sum(c.volume for c in candles[-30:]) / min(30, len(candles)),
            volatility=compute_volatility(candles),
            last_updated=datetime.now(UTC),
            candles=candles,
            technicals=compute_technicals(candles),
        )


class AlpacaProvider:
    name = "alpaca"

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

    async def get_snapshot(self, ticker: str) -> QuoteSnapshot:
        end = datetime.now(UTC)
        start = end - timedelta(days=2)
        bars_url = (
            f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
            f"?timeframe=1Min&start={start.isoformat()}&end={end.isoformat()}&limit=240"
        )
        quotes_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest"
        async with httpx.AsyncClient(timeout=10, headers=self.headers) as client:
            bars_resp, quote_resp = await asyncio.gather(client.get(bars_url), client.get(quotes_url))
        bars_resp.raise_for_status()
        quote_resp.raise_for_status()
        bars = bars_resp.json().get("bars", [])
        latest_quote = quote_resp.json().get("quote", {})
        if not bars:
            raise ValueError(f"No alpaca bars for {ticker}")
        candles = [
            Candle(
                timestamp=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
            )
            for item in bars
        ]
        last = candles[-1].close
        previous_close = candles[-2].close if len(candles) > 1 else last
        bid = to_optional_float(latest_quote.get("bp"))
        ask = to_optional_float(latest_quote.get("ap"))
        spread = ask - bid if bid is not None and ask is not None else None
        return QuoteSnapshot(
            ticker=ticker,
            source=self.name,
            last=last,
            previous_close=previous_close,
            change=last - previous_close,
            change_percent=((last - previous_close) / previous_close * 100) if previous_close else 0.0,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=candles[-1].volume,
            avg_volume=sum(c.volume for c in candles[-30:]) / min(30, len(candles)),
            volatility=compute_volatility(candles),
            last_updated=datetime.now(UTC),
            candles=candles,
            technicals=compute_technicals(candles),
        )


@dataclass(slots=True)
class CacheEntry:
    snapshot: QuoteSnapshot
    created_at: datetime


class MarketDataService:
    def __init__(self, config: Settings, predictor: PredictionEngine) -> None:
        self.config = config
        self.predictor = predictor
        self._cache: dict[str, CacheEntry] = {}
        self.providers = []
        if config.polygon_api_key:
            self.providers.append(PolygonProvider(config.polygon_api_key))
        if config.alpaca_api_key and config.alpaca_api_secret:
            self.providers.append(AlpacaProvider(config.alpaca_api_key, config.alpaca_api_secret))
        self.providers.append(YahooFinanceProvider(config))

    async def get_snapshot(self, ticker: str, force: bool = False) -> QuoteSnapshot:
        cached = self._cache.get(ticker)
        if (
            cached
            and not force
            and (datetime.now(UTC) - cached.created_at).total_seconds() < self.config.refresh_seconds
        ):
            return cached.snapshot

        last_error: Exception | None = None
        for provider in self.providers:
            try:
                snapshot = await provider.get_snapshot(ticker)
                snapshot.prediction = self.predictor.predict(ticker, snapshot.candles)
                self._cache[ticker] = CacheEntry(snapshot=snapshot, created_at=datetime.now(UTC))
                return snapshot
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(f"Unable to load market data for {ticker}: {last_error}")

    async def get_snapshots(self, tickers: list[str], force: bool = False) -> dict[str, QuoteSnapshot]:
        tasks = [self.get_snapshot(ticker, force=force) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: dict[str, QuoteSnapshot] = {}
        for ticker, result in zip(tickers, results, strict=False):
            if isinstance(result, QuoteSnapshot):
                snapshots[ticker] = result
        return snapshots


@dataclass(slots=True)
class NewsCacheEntry:
    created_at: datetime
    items: list[NewsItem]


class NewsService:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self._cache: dict[str, NewsCacheEntry] = {}

    async def get_news_for_ticker(self, ticker: str, force: bool = False) -> list[NewsItem]:
        cached = self._cache.get(ticker)
        if cached and not force and (datetime.now(UTC) - cached.created_at).total_seconds() < 300:
            return cached.items

        yahoo_items = await asyncio.to_thread(self._load_yahoo_news, ticker)
        rss_items = await asyncio.to_thread(self._load_google_rss, ticker)
        combined = self._dedupe(yahoo_items + rss_items)[: self.config.max_news_items]
        self._cache[ticker] = NewsCacheEntry(created_at=datetime.now(UTC), items=combined)
        return combined

    async def get_news_for_tickers(self, tickers: list[str], force: bool = False) -> dict[str, list[NewsItem]]:
        results = await asyncio.gather(
            *(self.get_news_for_ticker(ticker, force=force) for ticker in tickers),
            return_exceptions=True,
        )
        return {
            ticker: result
            for ticker, result in zip(tickers, results, strict=False)
            if not isinstance(result, Exception)
        }

    @staticmethod
    def _load_yahoo_news(ticker: str) -> list[NewsItem]:
        raw_items = getattr(yf.Ticker(ticker), "news", []) or []
        items: list[NewsItem] = []
        for item in raw_items:
            link = item.get("link") or ""
            source = item.get("publisher") or urlparse(link).netloc or "Yahoo Finance"
            timestamp = item.get("providerPublishTime")
            published_at = datetime.fromtimestamp(timestamp, tz=UTC) if timestamp else datetime.now(UTC)
            items.append(
                NewsItem(
                    ticker=ticker,
                    title=item.get("title") or f"{ticker} market update",
                    url=link,
                    source=source,
                    published_at=published_at,
                    summary=item.get("summary") or "",
                )
            )
        return items

    @staticmethod
    def _load_google_rss(ticker: str) -> list[NewsItem]:
        query = quote(f"{ticker} stock market")
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        )
        items: list[NewsItem] = []
        for entry in feed.entries[:8]:
            published_at = datetime.now(UTC)
            if entry.get("published"):
                try:
                    published_at = parsedate_to_datetime(entry.published).astimezone(UTC)
                except (TypeError, ValueError):
                    pass
            source = entry.get("source", {}).get("title") or urlparse(entry.link).netloc or "Google News"
            items.append(
                NewsItem(
                    ticker=ticker,
                    title=entry.title,
                    url=entry.link,
                    source=source,
                    published_at=published_at,
                    summary=entry.get("summary", ""),
                )
            )
        return items

    @staticmethod
    def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
        deduped: dict[str, NewsItem] = {}
        for item in sorted(items, key=lambda news: news.published_at, reverse=True):
            key = "".join(char for char in item.title.lower() if char.isalnum() or char.isspace()).strip()
            if key not in deduped:
                deduped[key] = item
        return list(deduped.values())


class SentimentEngine:
    POSITIVE_TERMS = {
        "beat": 1.1,
        "growth": 0.8,
        "upgrade": 1.1,
        "surge": 1.0,
        "record": 0.8,
        "profit": 0.9,
        "outperform": 1.1,
        "buyback": 0.8,
        "bullish": 1.0,
        "partnership": 0.7,
    }
    NEGATIVE_TERMS = {
        "miss": -1.1,
        "downgrade": -1.2,
        "lawsuit": -1.0,
        "fraud": -1.4,
        "plunge": -1.2,
        "warning": -0.8,
        "decline": -0.7,
        "layoff": -0.7,
        "bearish": -1.0,
        "probe": -0.8,
    }
    EVENT_TERMS = {
        "earnings",
        "guidance",
        "sec",
        "federal reserve",
        "rate",
        "acquisition",
        "merger",
        "dividend",
        "buyback",
        "cpi",
    }
    SENSATIONAL_TERMS = {
        "guaranteed",
        "shocking",
        "secret",
        "explode",
        "moon",
        "crash now",
        "insane",
        "must buy",
    }

    def __init__(self, config: Settings) -> None:
        self.config = config

    def enrich(self, items: list[NewsItem]) -> list[NewsItem]:
        title_counts = Counter(self._normalize_title(item.title) for item in items)
        enriched: list[NewsItem] = []
        for item in items:
            text = f"{item.title} {item.summary}".lower()
            score = 0.0
            score += sum(weight for term, weight in self.POSITIVE_TERMS.items() if term in text)
            score += sum(weight for term, weight in self.NEGATIVE_TERMS.items() if term in text)
            sentiment = "neutral"
            if score >= 0.7:
                sentiment = "positive"
            elif score <= -0.7:
                sentiment = "negative"

            title_key = self._normalize_title(item.title)
            credibility = self._credibility(text, item.url, title_counts[title_key])
            relevance = min(
                1.0,
                0.45
                + 0.15 * sum(1 for term in self.EVENT_TERMS if term in text)
                + 0.1 * int(item.ticker.lower() in text),
            )
            flags: list[str] = []
            if credibility < 0.45:
                flags.append("low-credibility")
            if any(term in text for term in self.SENSATIONAL_TERMS):
                flags.append("possible-rumor")
            if title_counts[title_key] == 1 and credibility < 0.7:
                flags.append("uncorroborated")
            if any(term in text for term in self.EVENT_TERMS):
                flags.append("market-moving")

            enriched.append(
                item.model_copy(
                    update={
                        "sentiment": sentiment,
                        "sentiment_score": score,
                        "credibility": credibility,
                        "relevance": relevance,
                        "flags": flags,
                    }
                )
            )
        return sorted(
            enriched,
            key=lambda item: (item.credibility * item.relevance, item.published_at),
            reverse=True,
        )

    def _credibility(self, text: str, url: str, corroboration_count: int) -> float:
        domain = urlparse(url).netloc.lower()
        credibility = 0.45
        if any(domain.endswith(trusted) for trusted in self.config.trusted_news_domains):
            credibility += 0.3
        if url.startswith("https://"):
            credibility += 0.05
        if corroboration_count > 1:
            credibility += 0.1
        if any(term in text for term in self.SENSATIONAL_TERMS):
            credibility -= 0.2
        return max(0.05, min(0.99, credibility))

    @staticmethod
    def _normalize_title(title: str) -> str:
        return "".join(char for char in title.lower() if char.isalnum() or char.isspace()).strip()


class RobinhoodConnector:
    def __init__(self, config: Settings) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.robinhood_username and self.config.robinhood_password)

    async def fetch_positions(self) -> list[dict]:
        return await asyncio.to_thread(self._fetch_positions_sync)

    def _fetch_positions_sync(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            import robin_stocks.robinhood as robinhood
        except ImportError:
            return []
        robinhood.authentication.login(
            username=self.config.robinhood_username,
            password=self.config.robinhood_password,
            store_session=False,
        )
        raw_positions = robinhood.account.build_holdings()
        return [
            {
                "ticker": ticker.upper(),
                "shares": float(payload.get("quantity", 0)),
                "avg_cost": float(payload.get("average_buy_price", 0)),
                "thesis": "Synced from Robinhood",
            }
            for ticker, payload in raw_positions.items()
        ]


class PlaidConnector:
    def __init__(self, config: Settings) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.plaid_client_id and self.config.plaid_secret and self.config.plaid_access_token)

    async def fetch_expenses(self) -> list[dict]:
        if not self.enabled:
            return []
        end_date = date.today()
        start_date = date.fromordinal(end_date.toordinal() - 30)
        payload = {
            "client_id": self.config.plaid_client_id,
            "secret": self.config.plaid_secret,
            "access_token": self.config.plaid_access_token,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{self.config.plaid_base_url}/transactions/get",
                json=payload,
            )
            response.raise_for_status()
            transactions = response.json().get("transactions", [])
        return [
            {
                "category": ", ".join(transaction.get("category") or ["other"]),
                "amount": float(transaction["amount"]),
                "merchant": transaction.get("merchant_name") or transaction.get("name") or "Plaid",
                "incurred_on": transaction["date"],
            }
            for transaction in transactions
        ]


class PortfolioService:
    def __init__(self, store: SQLiteStore, config: Settings) -> None:
        self.store = store
        self.robinhood = RobinhoodConnector(config)
        self.plaid = PlaidConnector(config)

    async def build_summary(
        self,
        user_id: str,
        profile: UserProfile,
        snapshots: dict[str, QuoteSnapshot],
    ) -> PortfolioSummary:
        manual_positions = self.store.list_positions(user_id)
        connector_positions = await self.robinhood.fetch_positions() if self.robinhood.enabled else []
        expenses = self.store.list_expenses(user_id)
        plaid_expenses = await self.plaid.fetch_expenses() if self.plaid.enabled else []

        merged_positions = self._merge_positions(manual_positions + connector_positions)
        positions: list[PortfolioPosition] = []
        total_value = profile.cash_balance
        total_cost = 0.0
        day_pnl = 0.0

        for row in merged_positions:
            ticker = row["ticker"]
            snapshot = snapshots.get(ticker)
            market_price = snapshot.last if snapshot else row["avg_cost"]
            market_value = row["shares"] * market_price
            pnl_unrealized = (market_price - row["avg_cost"]) * row["shares"]
            risk_score = self._risk_from_snapshot(snapshot)
            positions.append(
                PortfolioPosition(
                    ticker=ticker,
                    shares=row["shares"],
                    avg_cost=row["avg_cost"],
                    market_price=market_price,
                    market_value=market_value,
                    pnl_unrealized=pnl_unrealized,
                    allocation=0.0,
                    risk_score=risk_score,
                    thesis=row.get("thesis", ""),
                )
            )
            total_value += market_value
            total_cost += row["avg_cost"] * row["shares"]
            if snapshot:
                day_pnl += row["shares"] * snapshot.change

        if total_value > 0:
            positions = [
                position.model_copy(update={"allocation": position.market_value / total_value})
                for position in positions
            ]

        external_expenses = [
            self._expense_from_dict(payload)
            for payload in plaid_expenses
            if payload.get("amount", 0) > 0
        ]
        all_expenses = sorted(expenses + external_expenses, key=lambda item: item.incurred_on, reverse=True)
        expenses_month = sum(item.amount for item in all_expenses)
        concentration = max((position.allocation for position in positions), default=0.0)
        diversification_score = max(0.0, 100 - concentration * 100 - max(len(positions) - 1, 0) * 2)
        risk_score = (
            sum(position.risk_score * position.allocation for position in positions) * 100 if positions else 0.0
        )
        total_pnl = total_value - profile.cash_balance - total_cost
        connector_status = {
            "manual": "active",
            "robinhood": "configured" if self.robinhood.enabled else "optional",
            "plaid": "configured" if self.plaid.enabled else "optional",
        }
        return PortfolioSummary(
            total_value=total_value,
            cash=profile.cash_balance,
            day_pnl=day_pnl,
            total_pnl=total_pnl,
            risk_score=risk_score,
            diversification_score=diversification_score,
            expenses_month=expenses_month,
            positions=sorted(positions, key=lambda position: position.market_value, reverse=True),
            expenses=all_expenses[:12],
            connector_status=connector_status,
            budget_advice=self._budget_advice(profile.monthly_budget, expenses_month, positions),
        )

    @staticmethod
    def _merge_positions(rows: list[dict]) -> list[dict]:
        merged: dict[str, dict] = defaultdict(lambda: {"shares": 0.0, "avg_cost": 0.0, "thesis": ""})
        for row in rows:
            ticker = row["ticker"].upper()
            existing = merged[ticker]
            total_cost = existing["shares"] * existing["avg_cost"] + row["shares"] * row["avg_cost"]
            existing["shares"] += row["shares"]
            existing["avg_cost"] = total_cost / existing["shares"] if existing["shares"] else row["avg_cost"]
            existing["ticker"] = ticker
            existing["thesis"] = row.get("thesis") or existing.get("thesis", "")
        return list(merged.values())

    @staticmethod
    def _risk_from_snapshot(snapshot: QuoteSnapshot | None) -> float:
        if snapshot is None:
            return 0.35
        spread_component = min(0.4, ((snapshot.spread or 0) / snapshot.last) * 100) if snapshot.last else 0.0
        volatility_component = min(0.6, snapshot.volatility * 6)
        return max(0.05, min(0.95, spread_component + volatility_component))

    @staticmethod
    def _expense_from_dict(payload: dict) -> ExpenseItem:
        incurred_on = payload["incurred_on"]
        if isinstance(incurred_on, str):
            incurred_on = date.fromisoformat(incurred_on)
        return ExpenseItem(
            category=payload["category"],
            amount=float(payload["amount"]),
            merchant=payload.get("merchant", ""),
            incurred_on=incurred_on,
        )

    @staticmethod
    def _budget_advice(
        monthly_budget: float,
        expenses_month: float,
        positions: list[PortfolioPosition],
    ) -> list[str]:
        advice: list[str] = []
        utilization = (expenses_month / monthly_budget) if monthly_budget else 0.0
        if utilization > 0.85:
            advice.append("Spending is running hot versus your monthly budget. Reduce discretionary flow before adding risk.")
        elif utilization < 0.5:
            advice.append("Budget utilization is contained. You have room to deploy cash gradually rather than chasing one entry.")
        if positions and max(position.allocation for position in positions) > 0.35:
            advice.append("Portfolio concentration is elevated. One position is above 35% allocation and should be reviewed.")
        return advice


class PaperTradingService:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def execute_trade(self, user_id: str, ticker: str, side: str, shares: float, price: float) -> None:
        self.store.add_sim_trade(user_id, ticker, side, shares, price)

    def build_book(self, user_id: str, snapshots: dict[str, QuoteSnapshot]) -> SimulationBook:
        trades = self.store.list_sim_trades(user_id)
        aggregates: dict[str, dict] = defaultdict(lambda: {"net_shares": 0.0, "cost_basis": 0.0})
        for trade in trades:
            side_multiplier = 1 if trade.side.upper() == "BUY" else -1
            record = aggregates[trade.ticker]
            record["net_shares"] += side_multiplier * trade.shares
            record["cost_basis"] += side_multiplier * trade.shares * trade.price

        positions: list[SimulationPosition] = []
        equity = 0.0
        gross_exposure = 0.0
        for ticker, record in aggregates.items():
            if abs(record["net_shares"]) < 1e-9:
                continue
            snapshot = snapshots.get(ticker)
            market_price = snapshot.last if snapshot else 0.0
            avg_entry = record["cost_basis"] / record["net_shares"] if record["net_shares"] else 0.0
            exposure = record["net_shares"] * market_price
            pnl_unrealized = (market_price - avg_entry) * record["net_shares"]
            positions.append(
                SimulationPosition(
                    ticker=ticker,
                    net_shares=record["net_shares"],
                    avg_entry=avg_entry,
                    market_price=market_price,
                    exposure=exposure,
                    pnl_unrealized=pnl_unrealized,
                )
            )
            equity += pnl_unrealized
            gross_exposure += abs(exposure)

        return SimulationBook(
            equity=equity,
            gross_exposure=gross_exposure,
            positions=sorted(positions, key=lambda position: abs(position.exposure), reverse=True),
            trades=trades[-20:],
        )


class ScenarioEngine:
    def run(self, portfolio: PortfolioSummary, market_shock_pct: float) -> ScenarioResult:
        impacts: list[ScenarioImpact] = []
        estimated_portfolio_change = 0.0
        for position in portfolio.positions:
            shock_pct = market_shock_pct * (0.8 + position.risk_score * 0.6)
            estimated_pnl = position.market_value * (shock_pct / 100)
            impacts.append(
                ScenarioImpact(
                    ticker=position.ticker,
                    shock_pct=shock_pct,
                    estimated_pnl=estimated_pnl,
                )
            )
            estimated_portfolio_change += estimated_pnl

        base = portfolio.total_value or 1.0
        change_pct = estimated_portfolio_change / base * 100
        narrative = (
            f"A {market_shock_pct:.1f}% market move translates to an estimated "
            f"{change_pct:.2f}% portfolio swing after risk amplification."
        )
        return ScenarioResult(
            title="What-if shock test",
            market_shock_pct=market_shock_pct,
            estimated_portfolio_change=estimated_portfolio_change,
            estimated_change_pct=change_pct,
            impacts=sorted(impacts, key=lambda impact: abs(impact.estimated_pnl), reverse=True),
            narrative=narrative,
        )


class BrainService:
    def evaluate(
        self,
        profile: UserProfile,
        snapshots: dict[str, QuoteSnapshot],
        news_map: dict[str, list[NewsItem]],
        portfolio: PortfolioSummary,
    ) -> BrainDecision:
        scored: list[tuple[str, float, QuoteSnapshot, list[NewsItem], str, float, float]] = []
        opportunities: list[Insight] = []
        warnings: list[Insight] = []

        for ticker, snapshot in snapshots.items():
            related_news = news_map.get(ticker, [])
            sentiment_score = self._sentiment_score(related_news)
            risk_score = self._risk_score(snapshot)
            composite = self._composite_score(profile, snapshot, sentiment_score, risk_score)
            confidence = min(
                0.96,
                0.42
                + abs(snapshot.prediction.probability_up - 0.5) * 1.35
                + 0.05 * len(snapshot.technicals.patterns)
                + 0.08 * min(1.0, abs(sentiment_score)),
            )
            signal = "hold"
            if composite >= 0.28:
                signal = "buy"
            elif composite <= -0.28:
                signal = "trim"
            scored.append((ticker, composite, snapshot, related_news, signal, confidence, risk_score))

            if composite >= 0.28:
                opportunities.append(
                    Insight(
                        level="opportunity",
                        ticker=ticker,
                        headline=self._opportunity_headline(ticker, snapshot, related_news),
                        rationale=self._opportunity_rationale(snapshot, sentiment_score, profile),
                        risk_level=self._risk_label(risk_score),
                        confidence=confidence,
                        action=self._action_label(profile, snapshot),
                        horizon=self._horizon_label(profile),
                        drivers=self._drivers(snapshot, sentiment_score),
                        timestamp=datetime.now(UTC),
                    )
                )
            if risk_score >= 0.58 or composite <= -0.28:
                warnings.append(
                    Insight(
                        level="warning",
                        ticker=ticker,
                        headline=self._warning_headline(ticker, snapshot, related_news),
                        rationale=self._warning_rationale(snapshot, sentiment_score, profile),
                        risk_level=self._risk_label(risk_score),
                        confidence=confidence,
                        action="reduce size" if risk_score >= 0.58 else "wait",
                        horizon="now",
                        drivers=self._drivers(snapshot, sentiment_score),
                        timestamp=datetime.now(UTC),
                    )
                )

        scored.sort(key=lambda item: item[1], reverse=True)
        day_trades = [
            ticker
            for ticker, score, snapshot, *_ in scored
            if score > 0 and snapshot.technicals.volume_spike > 1.2 and (snapshot.spread or 0) <= snapshot.last * 0.004
        ][:3]
        long_term = [
            ticker
            for ticker, score, snapshot, *_ in scored
            if score > 0 and snapshot.technicals.trend == "bullish" and snapshot.volatility < 0.7
        ][:3]

        market_sentiment_score = sum(self._sentiment_score(items) for items in news_map.values()) / max(1, len(news_map))
        regime = self._market_regime(scored, market_sentiment_score)
        risk_posture = self._risk_posture(profile, portfolio)
        confidence = min(
            0.95,
            0.5 + sum(abs(score) for _, score, *_ in scored[:4]) / max(1, min(len(scored), 4)) * 0.22,
        )
        summary = self._summary_text(regime, profile, scored, portfolio)

        return BrainDecision(
            market_regime=regime,
            summary=summary,
            risk_posture=risk_posture,
            confidence=confidence,
            opportunities=opportunities[:5],
            warnings=warnings[:5],
            top_day_trade_tickers=day_trades,
            top_long_term_tickers=long_term,
            strategy_notes=self._strategy_notes(profile, portfolio, regime, scored),
            market_sentiment_score=market_sentiment_score,
        )

    @staticmethod
    def _sentiment_score(items: list[NewsItem]) -> float:
        if not items:
            return 0.0
        weighted = 0.0
        total_weight = 0.0
        for item in items:
            weighted += item.sentiment_score * max(0.1, item.credibility) * max(0.1, item.relevance)
            total_weight += max(0.1, item.credibility) * max(0.1, item.relevance)
        return weighted / total_weight if total_weight else 0.0

    def _composite_score(
        self,
        profile: UserProfile,
        snapshot: QuoteSnapshot,
        sentiment_score: float,
        risk_score: float,
    ) -> float:
        trend_bias = 0.18 if snapshot.technicals.trend == "bullish" else -0.18 if snapshot.technicals.trend == "bearish" else 0.0
        prediction_bias = (snapshot.prediction.probability_up - 0.5) * 1.1
        rsi_bias = (snapshot.technicals.rsi - 50) / 100
        volume_bias = min(0.22, max(-0.12, (snapshot.technicals.volume_spike - 1) * 0.16))
        pattern_bias = 0.05 * len(snapshot.technicals.patterns)
        risk_penalty = risk_score * 0.28
        horizon_bias = 0.0
        if profile.investment_horizon == "short-term":
            horizon_bias += volume_bias + prediction_bias * 0.2
        elif profile.investment_horizon == "long-term":
            horizon_bias += 0.12 if snapshot.volatility < 0.5 else -0.08
        return trend_bias + prediction_bias + rsi_bias + volume_bias + pattern_bias + sentiment_score * 0.08 + horizon_bias - risk_penalty

    @staticmethod
    def _risk_score(snapshot: QuoteSnapshot) -> float:
        spread_risk = min(0.25, ((snapshot.spread or 0) / snapshot.last) * 70) if snapshot.last else 0.0
        volatility_risk = min(0.7, snapshot.volatility * 0.9)
        return max(0.05, min(0.95, spread_risk + volatility_risk))

    @staticmethod
    def _risk_label(risk_score: float) -> str:
        if risk_score >= 0.66:
            return "high"
        if risk_score >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _horizon_label(profile: UserProfile) -> str:
        if profile.investment_horizon == "short-term":
            return "intraday"
        if profile.investment_horizon == "long-term":
            return "multi-month"
        return "swing"

    @staticmethod
    def _drivers(snapshot: QuoteSnapshot, sentiment_score: float) -> list[str]:
        drivers = [snapshot.technicals.trend, snapshot.prediction.label, f"sentiment:{sentiment_score:.2f}"]
        drivers.extend(snapshot.technicals.patterns[:2])
        return [driver for driver in drivers if driver]

    def _opportunity_headline(self, ticker: str, snapshot: QuoteSnapshot, news: list[NewsItem]) -> str:
        if snapshot.technicals.volume_spike > 1.8:
            return f"{ticker} is printing unusual volume with tradable momentum"
        if news and "market-moving" in news[0].flags:
            return f"{ticker} is reacting to a market-moving news catalyst"
        return f"{ticker} aligns with the current signal stack"

    def _opportunity_rationale(self, snapshot: QuoteSnapshot, sentiment_score: float, profile: UserProfile) -> str:
        return (
            f"Trend is {snapshot.technicals.trend}, RSI is {snapshot.technicals.rsi:.1f}, "
            f"model probability-up is {snapshot.prediction.probability_up:.0%}, "
            f"and weighted news sentiment is {sentiment_score:.2f}. "
            f"This fits a {profile.investment_horizon} horizon better than a blind momentum chase."
        )

    def _warning_headline(self, ticker: str, snapshot: QuoteSnapshot, news: list[NewsItem]) -> str:
        if snapshot.volatility > 0.7:
            return f"{ticker} volatility is elevated and position sizing should tighten"
        if news and any("possible-rumor" in item.flags for item in news):
            return f"{ticker} headline flow is noisy and credibility is mixed"
        return f"{ticker} carries an unfavorable risk/reward profile right now"

    def _warning_rationale(self, snapshot: QuoteSnapshot, sentiment_score: float, profile: UserProfile) -> str:
        spread_pct = ((snapshot.spread or 0) / snapshot.last * 100) if snapshot.last else 0.0
        return (
            f"Risk is rising from volatility ({snapshot.volatility:.2f}), spread "
            f"({spread_pct:.2f}%), and sentiment ({sentiment_score:.2f}). "
            f"That conflicts with a {profile.risk_tolerance} risk mandate."
        )

    def _action_label(self, profile: UserProfile, snapshot: QuoteSnapshot) -> str:
        if profile.investment_horizon == "short-term":
            return "watch breakout" if snapshot.technicals.volume_spike > 1.3 else "scalp only on confirmation"
        if profile.investment_horizon == "long-term":
            return "scale in"
        return "accumulate selectively"

    @staticmethod
    def _market_regime(scored, market_sentiment_score: float) -> str:
        if not scored:
            return "neutral"
        average_score = sum(score for _, score, *_ in scored[:5]) / min(len(scored), 5)
        if average_score > 0.18 and market_sentiment_score > -0.2:
            return "risk-on"
        if average_score < -0.1:
            return "defensive"
        return "range-bound"

    @staticmethod
    def _risk_posture(profile: UserProfile, portfolio: PortfolioSummary) -> str:
        if profile.risk_tolerance == "aggressive" and portfolio.cash > portfolio.total_value * 0.15:
            return "offensive"
        if portfolio.risk_score > 55 or profile.risk_tolerance == "conservative":
            return "capital-preservation"
        return "balanced"

    def _summary_text(self, regime: str, profile: UserProfile, scored, portfolio: PortfolioSummary) -> str:
        if not scored:
            return "Waiting on market data to form a view."
        top_ticker, _, top_snapshot, *_ = scored[0]
        return (
            f"Market regime is {regime}. The strongest aligned setup is {top_ticker} with "
            f"{top_snapshot.prediction.probability_up:.0%} probability-up and {top_snapshot.technicals.trend} trend. "
            f"Portfolio risk is {portfolio.risk_score:.1f}/100, so recommendations are tuned for a "
            f"{profile.risk_tolerance} risk profile."
        )

    def _strategy_notes(self, profile: UserProfile, portfolio: PortfolioSummary, regime: str, scored) -> list[str]:
        notes = [f"Use {regime} positioning rules: press high-conviction names only when signal and sentiment agree."]
        if portfolio.expenses_month > profile.monthly_budget:
            notes.append("Spending is above budget, so favor smaller entries and preserve cash.")
        if profile.investment_horizon == "short-term":
            notes.append("For day trading, prioritize names with volume spike above 1.3x and tight spreads.")
        if profile.investment_horizon == "long-term":
            notes.append("For long-term accumulation, focus on bullish-trend names and average in across sessions.")
        if scored:
            notes.append(f"Highest priority watchlist name: {scored[0][0]}.")
        return notes


class ChatService:
    TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def handle_message(
        self,
        user_id: str,
        message: str,
        profile: UserProfile,
        brain: BrainDecision,
        portfolio: PortfolioSummary,
        snapshots: dict[str, QuoteSnapshot],
        news_map: dict[str, list[NewsItem]],
    ) -> ChatReply:
        self.store.add_chat_message(user_id, "user", message)
        updated_profile = self._update_profile_from_message(profile, message)
        profile_changed = updated_profile != profile
        active_profile = updated_profile if profile_changed else profile
        if profile_changed:
            self.store.save_user_profile(updated_profile)
        if any(word in message.lower() for word in ("goal", "today", "budget", "focus", "need")):
            self.store.add_daily_goal(user_id, message)

        tickers = self._extract_tickers(message, active_profile.watchlist)
        response = self._compose_response(message, active_profile, brain, portfolio, snapshots, news_map, tickers)
        self.store.add_chat_message(user_id, "assistant", response)
        return ChatReply(
            response=response,
            suggestions=self.suggested_prompts(),
            referenced_tickers=tickers,
            updated_profile=updated_profile if profile_changed else None,
        )

    @staticmethod
    def suggested_prompts() -> list[str]:
        return [
            "What are the best day-trade setups right now?",
            "Review my portfolio risk and suggest adjustments.",
            "Which long-term names fit my current goal profile?",
            "Explain why the top ticker is being recommended.",
        ]

    def _compose_response(
        self,
        message: str,
        profile: UserProfile,
        brain: BrainDecision,
        portfolio: PortfolioSummary,
        snapshots: dict[str, QuoteSnapshot],
        news_map: dict[str, list[NewsItem]],
        tickers: list[str],
    ) -> str:
        lower = message.lower()
        if "portfolio" in lower or "risk" in lower:
            return (
                f"Portfolio value is ${portfolio.total_value:,.0f}, monthly expenses are ${portfolio.expenses_month:,.0f}, "
                f"and risk score is {portfolio.risk_score:.1f}/100. "
                f"Primary action: {brain.strategy_notes[0] if brain.strategy_notes else 'Hold risk steady until signals improve.'}"
            )
        if "day trade" in lower or "scalp" in lower:
            tickers_text = ", ".join(brain.top_day_trade_tickers or ["No clear setup"])
            return (
                f"Best day-trade candidates: {tickers_text}. "
                f"I am prioritizing names with volume expansion, tighter spreads, and bullish probability signals. "
                f"Current regime is {brain.market_regime}, so size smaller if volatility expands."
            )
        if "long term" in lower or "invest" in lower:
            tickers_text = ", ".join(brain.top_long_term_tickers or ["No strong long-term setup"])
            return (
                f"Best long-term candidates: {tickers_text}. "
                f"They rank highest because trend, model bias, and credible news flow align better than the rest of the watchlist."
            )
        if tickers:
            pieces = []
            for ticker in tickers[:3]:
                snapshot = snapshots.get(ticker)
                if not snapshot:
                    continue
                items = news_map.get(ticker, [])
                headline = items[0] if items else None
                headline_text = headline.title if headline else "no major headline catalyst"
                pieces.append(
                    f"{ticker} trades at ${snapshot.last:,.2f}, trend is {snapshot.technicals.trend}, "
                    f"model probability-up is {snapshot.prediction.probability_up:.0%}, "
                    f"risk is {snapshot.volatility:.2f}, and the lead catalyst is {headline_text}."
                )
            return " ".join(pieces) or brain.summary
        return (
            f"{brain.summary} Top day-trade tickers: {', '.join(brain.top_day_trade_tickers or ['none'])}. "
            f"Top long-term tickers: {', '.join(brain.top_long_term_tickers or ['none'])}. "
            f"Tell me if today's focus is income, capital preservation, or aggressive growth and I will retune the recommendations."
        )

    def _update_profile_from_message(self, profile: UserProfile, message: str) -> UserProfile:
        updated = profile.model_copy(deep=True)
        lower = message.lower()
        if "day trade" in lower or "short term" in lower:
            updated.investment_horizon = "short-term"
        if "long term" in lower or "retirement" in lower:
            updated.investment_horizon = "long-term"
        if "low risk" in lower or "conservative" in lower:
            updated.risk_tolerance = "conservative"
        if "aggressive" in lower or "high risk" in lower:
            updated.risk_tolerance = "aggressive"
        budget_match = re.search(r"\$?(\d{3,6})", message.replace(",", ""))
        if budget_match and "budget" in lower:
            updated.monthly_budget = float(budget_match.group(1))
        return updated

    def _extract_tickers(self, message: str, watchlist: list[str]) -> list[str]:
        mentioned = {match.group(0).upper() for match in self.TICKER_PATTERN.finditer(message)}
        return [ticker for ticker in watchlist if ticker in mentioned]


@dataclass(slots=True)
class DashboardState:
    profile: UserProfile
    market: dict[str, QuoteSnapshot] = field(default_factory=dict)
    news: dict[str, list[NewsItem]] = field(default_factory=dict)
    portfolio: PortfolioSummary = field(default_factory=PortfolioSummary)
    brain: BrainDecision = field(default_factory=BrainDecision)
    simulation: SimulationBook = field(default_factory=SimulationBook)
    scenario: ScenarioResult | None = None
    refreshed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PlatformOrchestrator:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.store = SQLiteStore(config.db_path)
        self.predictor = PredictionEngine()
        self.market_service = MarketDataService(config, self.predictor)
        self.news_service = NewsService(config)
        self.sentiment_engine = SentimentEngine(config)
        self.portfolio_service = PortfolioService(self.store, config)
        self.simulator = PaperTradingService(self.store)
        self.scenario_engine = ScenarioEngine()
        self.brain_service = BrainService()
        self.chat_service = ChatService(self.store)
        self._lock = asyncio.Lock()
        self._bootstrapped = False
        self.state: DashboardState | None = None

    async def ensure_bootstrapped(self) -> None:
        if self._bootstrapped:
            return
        self.store.initialize()
        profile = self.store.load_user_profile(self.config.default_user_id, self.config.watchlist)
        self.state = DashboardState(profile=profile)
        self._bootstrapped = True
        await self.refresh_all(force=True)

    async def ensure_fresh(self) -> None:
        await self.ensure_bootstrapped()
        assert self.state is not None
        age = (datetime.now(UTC) - self.state.refreshed_at).total_seconds()
        if age >= self.config.refresh_seconds:
            await self.refresh_all()

    async def refresh_all(self, force: bool = False) -> DashboardState:
        await self.ensure_bootstrapped()
        async with self._lock:
            profile = self.store.load_user_profile(self.config.default_user_id, self.config.watchlist)
            market, news = await asyncio.gather(
                self.market_service.get_snapshots(profile.watchlist, force=force),
                self.news_service.get_news_for_tickers(profile.watchlist[:6], force=force),
            )
            enriched_news = {ticker: self.sentiment_engine.enrich(items) for ticker, items in news.items()}
            portfolio = await self.portfolio_service.build_summary(profile.user_id, profile, market)
            simulation = self.simulator.build_book(profile.user_id, market)
            brain = self.brain_service.evaluate(profile, market, enriched_news, portfolio)
            scenario = self.scenario_engine.run(portfolio, market_shock_pct=-5.0)
            self.state = DashboardState(
                profile=profile,
                market=market,
                news=enriched_news,
                portfolio=portfolio,
                brain=brain,
                simulation=simulation,
                scenario=scenario,
                refreshed_at=datetime.now(UTC),
            )
            return self.state

    def get_state(self) -> DashboardState:
        if self.state is None:
            raise RuntimeError("Platform has not been bootstrapped")
        return self.state

    async def update_profile(
        self,
        investment_horizon: str,
        risk_tolerance: str,
        monthly_budget: float,
        cash_balance: float,
        watchlist: list[str],
    ) -> DashboardState:
        await self.ensure_bootstrapped()
        profile = self.store.load_user_profile(self.config.default_user_id, self.config.watchlist)
        updated = profile.model_copy(
            update={
                "investment_horizon": investment_horizon,
                "risk_tolerance": risk_tolerance,
                "monthly_budget": monthly_budget,
                "cash_balance": cash_balance,
                "watchlist": watchlist,
            }
        )
        self.store.save_user_profile(updated)
        return await self.refresh_all(force=True)

    async def add_position(self, ticker: str, shares: float, avg_cost: float, thesis: str = "") -> DashboardState:
        await self.ensure_bootstrapped()
        self.store.upsert_position(self.config.default_user_id, ticker, shares, avg_cost, thesis)
        return await self.refresh_all(force=True)

    async def add_expense(self, category: str, amount: float, merchant: str) -> DashboardState:
        await self.ensure_bootstrapped()
        self.store.add_expense(self.config.default_user_id, category, amount, merchant)
        return await self.refresh_all(force=True)

    async def execute_sim_trade(self, ticker: str, side: str, shares: float) -> DashboardState:
        await self.ensure_bootstrapped()
        state = self.get_state()
        snapshot = state.market.get(ticker)
        if snapshot is None:
            await self.refresh_all(force=True)
            state = self.get_state()
            snapshot = state.market.get(ticker)
        if snapshot is None:
            raise ValueError(f"Ticker {ticker} is not available")
        self.simulator.execute_trade(self.config.default_user_id, ticker, side, shares, snapshot.last)
        return await self.refresh_all(force=True)

    async def run_scenario(self, market_shock_pct: float) -> DashboardState:
        await self.ensure_fresh()
        state = self.get_state()
        scenario = self.scenario_engine.run(state.portfolio, market_shock_pct)
        self.state = DashboardState(
            profile=state.profile,
            market=state.market,
            news=state.news,
            portfolio=state.portfolio,
            brain=state.brain,
            simulation=state.simulation,
            scenario=scenario,
            refreshed_at=state.refreshed_at,
        )
        return self.state

    async def chat(self, message: str) -> tuple[ChatReply, DashboardState]:
        await self.ensure_fresh()
        state = self.get_state()
        reply = self.chat_service.handle_message(
            self.config.default_user_id,
            message,
            state.profile,
            state.brain,
            state.portfolio,
            state.market,
            state.news,
        )
        refreshed = await self.refresh_all()
        return reply, refreshed


platform = PlatformOrchestrator(settings)


def inject_theme() -> None:
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Unbounded:wght@500;700&display=swap" rel="stylesheet">
        """
    )
    ui.add_css(
        """
        :root {
          --panel: rgba(13, 17, 35, 0.72);
          --panel-strong: rgba(16, 22, 44, 0.88);
          --line: rgba(147, 220, 255, 0.18);
          --text: #edf5ff;
          --muted: #9cb2d6;
          --cyan: #4cf3ff;
          --green: #6dffac;
          --gold: #ffc857;
          --pink: #ff5dd3;
          --danger: #ff6b8a;
          --shadow: 0 25px 60px rgba(0, 0, 0, 0.36);
        }
        body {
          margin: 0;
          color: var(--text);
          font-family: 'Space Grotesk', sans-serif;
          background:
            radial-gradient(circle at top left, rgba(76, 243, 255, 0.14), transparent 24rem),
            radial-gradient(circle at top right, rgba(255, 93, 211, 0.12), transparent 22rem),
            linear-gradient(180deg, #040612 0%, #070b18 100%);
        }
        .page-shell {
          width: min(1440px, calc(100% - 2rem));
          margin: 0 auto;
          padding: 1.25rem 0 7rem;
          gap: 1rem;
        }
        .hero-card, .glass-card, .metric-card, .chat-card {
          background: var(--panel);
          border: 1px solid var(--line);
          backdrop-filter: blur(18px);
          box-shadow: var(--shadow);
          border-radius: 24px;
        }
        .hero-card {
          padding: 1.4rem;
          background:
            linear-gradient(135deg, rgba(76, 243, 255, 0.08), rgba(255, 93, 211, 0.08)),
            var(--panel-strong);
        }
        .glass-card { padding: 1rem; }
        .metric-card { padding: 1rem; min-height: 112px; }
        .metric-title, .subtle { color: var(--muted); }
        .title-font { font-family: 'Unbounded', sans-serif; }
        .neon-pill {
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 0.35rem 0.7rem;
          background: rgba(255, 255, 255, 0.04);
          color: var(--cyan);
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .positive { color: var(--green); }
        .negative { color: var(--danger); }
        .neutral { color: var(--gold); }
        .ai-badge {
          background: linear-gradient(135deg, rgba(76, 243, 255, 0.16), rgba(255, 93, 211, 0.18));
          border-radius: 16px;
          padding: 0.65rem 0.8rem;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .news-item, .insight-item, .chat-bubble {
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 18px;
          padding: 0.8rem;
          background: rgba(255, 255, 255, 0.03);
        }
        .chat-card {
          width: 380px;
          max-width: calc(100vw - 1rem);
          padding: 1rem;
        }
        .typing { animation: pulse 1.1s ease-in-out infinite; }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        @media (max-width: 1100px) {
          .chat-card {
            position: static !important;
            width: 100%;
          }
        }
        """
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def percent(value: float) -> str:
    return f"{value:.2f}%"


def signed_percent(value: float) -> str:
    return f"{value:+.2f}%"


def signed_money(value: float) -> str:
    return f"{value:+,.2f}"


def color_class(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def serialize_state() -> dict:
    state = platform.get_state()
    return {
        "profile": state.profile.model_dump(),
        "market": {ticker: snapshot.model_dump() for ticker, snapshot in state.market.items()},
        "news": {ticker: [item.model_dump() for item in items] for ticker, items in state.news.items()},
        "portfolio": state.portfolio.model_dump(),
        "brain": state.brain.model_dump(),
        "simulation": state.simulation.model_dump(),
        "scenario": state.scenario.model_dump() if state.scenario else None,
        "refreshed_at": state.refreshed_at.isoformat(),
    }


def price_figure(snapshot: QuoteSnapshot) -> go.Figure:
    figure = go.Figure()
    candles = snapshot.candles[-90:]
    if candles:
        x = [candle.timestamp for candle in candles]
        figure.add_trace(
            go.Candlestick(
                x=x,
                open=[candle.open for candle in candles],
                high=[candle.high for candle in candles],
                low=[candle.low for candle in candles],
                close=[candle.close for candle in candles],
                name=snapshot.ticker,
                increasing_line_color="#6dffac",
                decreasing_line_color="#ff6b8a",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[snapshot.technicals.sma_20] * len(x),
                mode="lines",
                line={"color": "#4cf3ff", "width": 1.2, "dash": "dot"},
                name="SMA20 proxy",
            )
        )
    figure.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf5ff", "family": "Space Grotesk"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        showlegend=False,
        height=360,
    )
    return figure


def risk_heatmap_figure(state: DashboardState) -> go.Figure:
    positions = state.portfolio.positions
    figure = go.Figure()
    if positions:
        figure.add_trace(
            go.Bar(
                x=[position.ticker for position in positions],
                y=[position.risk_score * 100 for position in positions],
                marker={"color": [position.allocation for position in positions], "colorscale": "Turbo"},
                text=[f"{position.allocation:.0%}" for position in positions],
                textposition="outside",
            )
        )
    figure.update_layout(
        title="Portfolio Risk Heatmap",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf5ff", "family": "Space Grotesk"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)", "title": "Risk score"},
        height=310,
    )
    return figure


def simulation_figure(state: DashboardState) -> go.Figure:
    figure = go.Figure()
    positions = state.simulation.positions
    if positions:
        figure.add_trace(
            go.Bar(
                x=[position.ticker for position in positions],
                y=[position.pnl_unrealized for position in positions],
                marker_color=[
                    "#6dffac" if position.pnl_unrealized >= 0 else "#ff6b8a"
                    for position in positions
                ],
            )
        )
    figure.update_layout(
        title="Paper Trading PnL",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf5ff", "family": "Space Grotesk"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        height=280,
    )
    return figure


@app.get("/api/state")
async def api_state():
    await platform.ensure_fresh()
    return jsonable_encoder(serialize_state())


@app.get("/api/market")
async def api_market():
    await platform.ensure_fresh()
    return jsonable_encoder({ticker: snapshot.model_dump() for ticker, snapshot in platform.get_state().market.items()})


@app.get("/api/brain")
async def api_brain():
    await platform.ensure_fresh()
    return jsonable_encoder(platform.get_state().brain.model_dump())


@app.post("/api/chat")
async def api_chat(payload: dict):
    reply, _ = await platform.chat(payload.get("message", ""))
    return jsonable_encoder(reply.model_dump())


@app.post("/api/portfolio/positions")
async def api_add_position(request: PositionRequest):
    state = await platform.add_position(request.ticker, request.shares, request.avg_cost, request.thesis)
    return jsonable_encoder({"portfolio": state.portfolio.model_dump()})


@app.post("/api/portfolio/expenses")
async def api_add_expense(request: ExpenseRequest):
    state = await platform.add_expense(request.category, request.amount, request.merchant)
    return jsonable_encoder({"portfolio": state.portfolio.model_dump()})


@ui.page("/")
async def home() -> None:
    inject_theme()
    ui.dark_mode().enable()
    await platform.ensure_bootstrapped()
    state = platform.get_state()
    selected_ticker = {"value": state.profile.watchlist[0] if state.profile.watchlist else "AAPL"}

    def refresh_all_sections() -> None:
        hero_metrics.refresh()
        market_strip.refresh()
        chart_panel.refresh()
        ai_panel.refresh()
        portfolio_panel.refresh()
        news_panel.refresh()
        simulator_panel.refresh()
        chat_history.refresh()

    async def timed_refresh() -> None:
        await platform.refresh_all()
        refresh_all_sections()

    async def save_profile() -> None:
        watchlist = [ticker.strip().upper() for ticker in watchlist_input.value.split(",") if ticker.strip()]
        await platform.update_profile(
            horizon_select.value,
            risk_select.value,
            float(budget_input.value or 0),
            float(cash_input.value or 0),
            watchlist or settings.watchlist,
        )
        refresh_all_sections()
        ui.notify("Profile updated")

    async def save_position() -> None:
        await platform.add_position(
            position_ticker.value,
            float(position_shares.value or 0),
            float(position_cost.value or 0),
            position_thesis.value or "",
        )
        refresh_all_sections()
        ui.notify("Position saved")

    async def save_expense() -> None:
        await platform.add_expense(
            expense_category.value,
            float(expense_amount.value or 0),
            expense_merchant.value or "",
        )
        refresh_all_sections()
        ui.notify("Expense logged")

    async def execute_trade(side: str) -> None:
        await platform.execute_sim_trade(sim_ticker.value.upper(), side, float(sim_shares.value or 0))
        refresh_all_sections()
        ui.notify(f"Paper trade submitted: {side}")

    async def update_scenario() -> None:
        await platform.run_scenario(float(scenario_slider.value))
        simulator_panel.refresh()

    async def send_chat(prompt: str | None = None) -> None:
        message = (prompt or chat_input.value or "").strip()
        if not message:
            return
        typing_note.set_visibility(True)
        reply, _ = await platform.chat(message)
        chat_input.set_value("")
        typing_note.set_visibility(False)
        refresh_all_sections()
        ui.notify("AI insight updated")
        if reply.updated_profile is not None:
            current_state = platform.get_state()
            horizon_select.set_value(current_state.profile.investment_horizon)
            risk_select.set_value(current_state.profile.risk_tolerance)
            budget_input.set_value(current_state.profile.monthly_budget)
            cash_input.set_value(current_state.profile.cash_balance)
            watchlist_input.set_value(", ".join(current_state.profile.watchlist))

    with ui.column().classes("page-shell"):
        with ui.row().classes("w-full items-center justify-between gap-4 hero-card"):
            with ui.column().classes("gap-2"):
                ui.label("Financial Intelligence Platform").classes("neon-pill")
                ui.label("Neon FinBrain").classes("title-font text-4xl")
                ui.label(
                    "A real-time AI market brain with no-key default data feeds, local ML, news credibility scoring, paper trading, and portfolio reasoning."
                ).classes("subtle max-w-3xl")
            with ui.column().classes("items-end gap-2"):
                ui.label("No-key mode: Yahoo Finance + free RSS/news").classes("neon-pill")
                ui.label("Optional connectors: Robinhood, Plaid, Polygon, Alpaca").classes("subtle")

        @ui.refreshable
        def hero_metrics() -> None:
            current = platform.get_state()
            with ui.grid(columns=5).classes("w-full gap-4"):
                metrics = [
                    ("Total Value", money(current.portfolio.total_value), "metric-card"),
                    ("Day PnL", money(current.portfolio.day_pnl), f"metric-card {color_class(current.portfolio.day_pnl)}"),
                    ("Market Regime", current.brain.market_regime, "metric-card"),
                    ("Brain Confidence", f"{current.brain.confidence:.0%}", "metric-card"),
                    ("Last Refresh", current.refreshed_at.strftime("%H:%M:%S"), "metric-card"),
                ]
                for title, value, classes in metrics:
                    with ui.card().classes(classes):
                        ui.label(title).classes("metric-title")
                        ui.label(str(value)).classes("text-2xl font-bold")

        @ui.refreshable
        def market_strip() -> None:
            current = platform.get_state()
            with ui.grid(columns=4).classes("w-full gap-4"):
                for ticker in current.profile.watchlist[:8]:
                    snapshot = current.market.get(ticker)
                    if snapshot is None:
                        continue
                    with ui.card().classes("glass-card"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(ticker).classes("text-lg font-bold")
                            ui.label(snapshot.source).classes("neon-pill")
                        ui.label(money(snapshot.last)).classes("text-2xl font-bold")
                        ui.label(signed_percent(snapshot.change_percent)).classes(color_class(snapshot.change_percent))
                        ui.label(
                            f"RSI {snapshot.technicals.rsi:.1f} | Vol spike {snapshot.technicals.volume_spike:.2f}x | Trend {snapshot.technicals.trend}"
                        ).classes("subtle")

        @ui.refreshable
        def chart_panel() -> None:
            current = platform.get_state()
            watchlist = current.profile.watchlist or settings.watchlist
            if selected_ticker["value"] not in watchlist:
                selected_ticker["value"] = watchlist[0]
            snapshot = current.market.get(selected_ticker["value"])
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Live Market Engine").classes("title-font text-xl")
                    ui.select(
                        options=watchlist,
                        value=selected_ticker["value"],
                        on_change=lambda event: (selected_ticker.__setitem__("value", event.value), chart_panel.refresh()),
                    ).classes("w-48")
                    if snapshot is not None:
                        ui.plotly(price_figure(snapshot)).classes("w-full")
                with ui.card().classes("glass-card min-w-[320px]"):
                    ui.label("Ticker Intelligence").classes("title-font text-xl")
                    if snapshot is not None:
                        with ui.column().classes("w-full gap-3"):
                            for label, value in [
                                ("Prediction", f"{snapshot.prediction.label} ({snapshot.prediction.probability_up:.0%} up)"),
                                ("Expected Move", percent(snapshot.prediction.expected_move_pct)),
                                ("Spread", money(snapshot.spread or 0)),
                                ("Volatility", f"{snapshot.volatility:.2f}"),
                                ("Patterns", ", ".join(snapshot.technicals.patterns or ['none'])),
                            ]:
                                with ui.row().classes("w-full items-center justify-between ai-badge"):
                                    ui.label(label).classes("subtle")
                                    ui.label(value).classes("font-semibold")

        @ui.refreshable
        def ai_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("AI Brain").classes("title-font text-xl")
                    ui.label(current.brain.summary).classes("subtle")
                    ui.linear_progress(value=current.brain.confidence, color="cyan").classes("mt-2")
                    ui.label(f"Confidence {current.brain.confidence:.0%} | Sentiment score {current.brain.market_sentiment_score:.2f}").classes("subtle")
                    ui.separator()
                    ui.label("Autonomous insights").classes("font-semibold")
                    for insight in current.brain.opportunities[:3]:
                        with ui.column().classes("insight-item"):
                            ui.label(f"{insight.ticker}: {insight.headline}").classes("font-semibold")
                            ui.label(insight.rationale).classes("subtle")
                            ui.label(f"Action: {insight.action} | Risk: {insight.risk_level} | Confidence: {insight.confidence:.0%}").classes("subtle")
                with ui.card().classes("glass-card w-full"):
                    ui.label("Risk warnings").classes("title-font text-xl")
                    for warning in current.brain.warnings[:3]:
                        with ui.column().classes("insight-item"):
                            ui.label(f"{warning.ticker}: {warning.headline}").classes("font-semibold negative")
                            ui.label(warning.rationale).classes("subtle")
                    ui.separator()
                    ui.label("AI-generated strategy").classes("font-semibold")
                    for note in current.brain.strategy_notes:
                        ui.label(f"• {note}").classes("subtle")

        @ui.refreshable
        def portfolio_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Portfolio Tracker").classes("title-font text-xl")
                    ui.label(
                        f"Cash {money(current.portfolio.cash)} | Total PnL {signed_money(current.portfolio.total_pnl)} | Expenses this month {money(current.portfolio.expenses_month)}"
                    ).classes("subtle")
                    if current.portfolio.positions:
                        for position in current.portfolio.positions[:8]:
                            with ui.row().classes("w-full items-center justify-between ai-badge"):
                                ui.label(f"{position.ticker} {position.shares:.2f} sh")
                                ui.label(
                                    f"{money(position.market_value)} | {signed_money(position.pnl_unrealized)} | alloc {position.allocation:.0%}"
                                ).classes(color_class(position.pnl_unrealized))
                    else:
                        ui.label("No live positions stored yet. Add manual positions below or wire an optional connector.").classes("subtle")
                    ui.separator()
                    for advice in current.portfolio.budget_advice:
                        ui.label(f"• {advice}").classes("subtle")
                with ui.card().classes("glass-card w-full"):
                    ui.plotly(risk_heatmap_figure(current)).classes("w-full")

        @ui.refreshable
        def news_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("News + Sentiment Engine").classes("title-font text-xl")
                    for ticker, items in list(current.news.items())[:4]:
                        ui.label(ticker).classes("font-semibold mt-2")
                        for item in items[:2]:
                            with ui.column().classes("news-item mb-2"):
                                with ui.row().classes("w-full items-center justify-between"):
                                    ui.link(item.title, item.url, new_tab=True).classes("font-semibold")
                                    ui.label(item.sentiment).classes(color_class(item.sentiment_score))
                                ui.label(
                                    f"{item.source} | credibility {item.credibility:.0%} | relevance {item.relevance:.0%}"
                                ).classes("subtle")
                                if item.flags:
                                    ui.label("Flags: " + ", ".join(item.flags)).classes("subtle")

        @ui.refreshable
        def simulator_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Paper Trading Simulator").classes("title-font text-xl")
                    ui.label(
                        f"Equity {signed_money(current.simulation.equity)} | Gross exposure {money(current.simulation.gross_exposure)}"
                    ).classes("subtle")
                    ui.plotly(simulation_figure(current)).classes("w-full")
                with ui.card().classes("glass-card w-full"):
                    ui.label("What-if Scenario Engine").classes("title-font text-xl")
                    ui.label(current.scenario.narrative if current.scenario else "Run a stress test").classes("subtle")
                    if current.scenario:
                        for impact in current.scenario.impacts[:5]:
                            with ui.row().classes("w-full items-center justify-between ai-badge"):
                                ui.label(f"{impact.ticker} shock {signed_percent(impact.shock_pct)}")
                                ui.label(signed_money(impact.estimated_pnl)).classes(color_class(impact.estimated_pnl))

        hero_metrics()
        market_strip()
        chart_panel()
        ai_panel()
        portfolio_panel()
        news_panel()
        simulator_panel()

        with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
            with ui.card().classes("glass-card w-full"):
                ui.label("Profile & Goal Controls").classes("title-font text-xl")
                with ui.row().classes("w-full gap-3"):
                    horizon_select = ui.select(
                        options=["short-term", "balanced", "long-term"],
                        value=state.profile.investment_horizon,
                        label="Horizon",
                    ).classes("w-48")
                    risk_select = ui.select(
                        options=["conservative", "moderate", "aggressive"],
                        value=state.profile.risk_tolerance,
                        label="Risk",
                    ).classes("w-48")
                    budget_input = ui.number("Monthly budget", value=state.profile.monthly_budget).classes("w-40")
                    cash_input = ui.number("Cash", value=state.profile.cash_balance).classes("w-40")
                watchlist_input = ui.input("Watchlist CSV", value=", ".join(state.profile.watchlist)).classes("w-full")
                ui.button("Save profile", on_click=save_profile)

            with ui.card().classes("glass-card w-full"):
                ui.label("Manual Portfolio Input").classes("title-font text-xl")
                with ui.row().classes("w-full gap-3"):
                    position_ticker = ui.input("Ticker", value="AAPL").classes("w-24")
                    position_shares = ui.number("Shares", value=10).classes("w-28")
                    position_cost = ui.number("Avg cost", value=180).classes("w-32")
                    position_thesis = ui.input("Thesis", value="Core quality long").classes("grow")
                ui.button("Save position", on_click=save_position)
                ui.separator()
                ui.label("Expense Logger").classes("font-semibold")
                with ui.row().classes("w-full gap-3"):
                    expense_category = ui.input("Category", value="Dining").classes("w-40")
                    expense_amount = ui.number("Amount", value=45).classes("w-28")
                    expense_merchant = ui.input("Merchant", value="Sample merchant").classes("grow")
                ui.button("Log expense", on_click=save_expense)

            with ui.card().classes("glass-card w-full"):
                ui.label("Paper Trade Controls").classes("title-font text-xl")
                with ui.row().classes("w-full gap-3"):
                    sim_ticker = ui.input("Ticker", value=selected_ticker["value"]).classes("w-24")
                    sim_shares = ui.number("Shares", value=5).classes("w-28")
                with ui.row().classes("gap-3"):
                    ui.button("Sim BUY", on_click=lambda: asyncio.create_task(execute_trade("BUY")))
                    ui.button("Sim SELL", on_click=lambda: asyncio.create_task(execute_trade("SELL")))
                ui.separator()
                ui.label("Scenario shock").classes("font-semibold")
                scenario_slider = ui.slider(min=-15, max=15, value=-5, step=1).classes("w-full")
                ui.button("Run scenario", on_click=update_scenario)

    chat_container = ui.card().classes("chat-card fixed bottom-4 right-4 z-50")
    with chat_container:
        ui.label("AI Assistant").classes("title-font text-xl")
        ui.label("Context-aware portfolio and market copilot").classes("subtle")

        @ui.refreshable
        def chat_history() -> None:
            for message in platform.store.list_recent_chat_messages(settings.default_user_id, limit=10):
                css = "chat-bubble"
                if message.role == "assistant":
                    css += " border-cyan-400"
                with ui.column().classes(css):
                    ui.label(message.role.upper()).classes("metric-title text-xs")
                    ui.label(message.message).classes("subtle")
                    ui.label(message.timestamp.strftime("%H:%M:%S")).classes("metric-title text-xs")

        chat_history()
        typing_note = ui.label("AI is reasoning...").classes("typing subtle")
        typing_note.set_visibility(False)
        chat_input = ui.input("Ask about goals, tickers, risk, or strategy").classes("w-full")
        with ui.row().classes("w-full gap-2"):
            ui.button("Send", on_click=lambda: asyncio.create_task(send_chat()))
        ui.label("Suggested prompts").classes("font-semibold mt-2")
        for suggestion in platform.chat_service.suggested_prompts():
            ui.button(
                suggestion,
                on_click=lambda _, prompt=suggestion: asyncio.create_task(send_chat(prompt)),
            ).props("flat").classes("w-full")

    ui.timer(settings.refresh_seconds, lambda: asyncio.create_task(timed_refresh()))


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=settings.app_name,
        reload=False,
        favicon="💹",
        host="0.0.0.0",
        port=8080,
    )
