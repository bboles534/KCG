"""Alfa Stock AI single-file edition.

Install:
    pip install nicegui fastapi httpx numpy pandas plotly scikit-learn yfinance feedparser pydantic

Run:
    python3 neon_finbrain.py

"""

from __future__ import annotations

import asyncio
import math
import os
import re
import sqlite3
import hashlib
import hmac
import secrets
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import time
from typing import Any
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
def local_time_text(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone().strftime("%I:%M:%S %p")



@dataclass(slots=True)
class Settings:
    app_name: str = "Alfa Stock AI"
    refresh_seconds: float = max(1.0, min(5.0, float(os.getenv("REFRESH_SECONDS", "3"))))
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
    plaid_mock: bool = os.getenv("PLAID_MOCK", "true").lower() == "true"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "demo-user")
    auth_secret_key: str = os.getenv("AUTH_SECRET_KEY", os.getenv("SECRET_KEY", "dev-auth-secret-change-this"))
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


class ClosedTrade(BaseModel):
    """A matched BUY → SELL pair with realized P&L."""
    ticker: str
    shares: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    closed_at: datetime


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
    starting_cash: float = 10000.0
    cash_available: float = 10000.0
    cash_deployed: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_value: float = 10000.0
    return_pct: float = 0.0
    positions: list[SimulationPosition] = Field(default_factory=list)
    trades: list[SimulatedTrade] = Field(default_factory=list)
    closed_trades: list[ClosedTrade] = Field(default_factory=list)


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
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
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

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260000)
        return f"pbkdf2_sha256${salt}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        try:
            algorithm, salt, stored_digest = password_hash.split("$", 2)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260000).hex()
            return hmac.compare_digest(digest, stored_digest)
        except Exception:
            return False

    def create_user(self, full_name: str, email: str, password: str) -> tuple[bool, str]:
        email = email.strip().lower()
        full_name = full_name.strip()
        if not full_name or not email or not password:
            return False, "Please fill in all fields."
        if len(password) < 8:
            return False, "Password must be at least 8 characters."
        with self._connect() as connection:
            existing = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                return False, "An account with that email already exists."
            connection.execute(
                "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (full_name, email, self._hash_password(password), datetime.utcnow().isoformat()),
            )
        return True, "Account created successfully."

    def authenticate_user(self, email: str, password: str) -> dict[str, Any] | None:
        email = email.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row and self._verify_password(password, row["password_hash"]):
            return {"id": row["id"], "full_name": row["full_name"], "email": row["email"], "user_id": row["email"]}
        return None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        email = email.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, full_name, email FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if not row:
            return None
        return {"id": row["id"], "full_name": row["full_name"], "email": row["email"], "user_id": row["email"]}

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
            last_updated=datetime.now(timezone.utc),
            candles=candles,
            technicals=compute_technicals(candles),
        )

    @staticmethod
    def _to_candles(frame: pd.DataFrame) -> list[Candle]:
        candles: list[Candle] = []
        for timestamp, row in frame.iterrows():
            ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
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
        end = datetime.now(timezone.utc)
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
                timestamp=datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc),
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
            last_updated=datetime.now(timezone.utc),
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
        end = datetime.now(timezone.utc)
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
            last_updated=datetime.now(timezone.utc),
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
            and (datetime.now(timezone.utc) - cached.created_at).total_seconds() < self.config.refresh_seconds
        ):
            return cached.snapshot

        last_error: Exception | None = None
        for provider in self.providers:
            try:
                snapshot = await provider.get_snapshot(ticker)
                snapshot.prediction = self.predictor.predict(ticker, snapshot.candles)
                self._cache[ticker] = CacheEntry(snapshot=snapshot, created_at=datetime.now(timezone.utc))
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
        if cached and not force and (datetime.now(timezone.utc) - cached.created_at).total_seconds() < max(5.0, self.config.refresh_seconds):
            return cached.items

        yahoo_items = await asyncio.to_thread(self._load_yahoo_news, ticker)
        rss_items = await asyncio.to_thread(self._load_google_rss, ticker)
        combined = self._dedupe(yahoo_items + rss_items)[: self.config.max_news_items]
        self._cache[ticker] = NewsCacheEntry(created_at=datetime.now(timezone.utc), items=combined)
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
            published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else datetime.now(timezone.utc)
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
            published_at = datetime.now(timezone.utc)
            if entry.get("published"):
                try:
                    published_at = parsedate_to_datetime(entry.published).astimezone(timezone.utc)
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


@dataclass
class RobinhoodSyncResult:
    ok: bool
    positions_synced: int = 0
    error: str = ""
    synced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PlaidSyncResult:
    ok: bool
    transactions_synced: int = 0
    accounts_found: int = 0
    error: str = ""
    synced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RobinhoodConnector:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self._last_sync: RobinhoodSyncResult | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.robinhood_username and self.config.robinhood_password)

    @property
    def last_sync(self) -> RobinhoodSyncResult | None:
        return self._last_sync

    async def fetch_positions(self) -> list[dict]:
        return await asyncio.to_thread(self._fetch_positions_sync)

    async def sync(self) -> RobinhoodSyncResult:
        positions, result = await asyncio.to_thread(self._sync_sync)
        self._last_sync = result
        return result

    def _login(self, robinhood: Any) -> None:
        robinhood.authentication.login(
            username=self.config.robinhood_username,
            password=self.config.robinhood_password,
            store_session=True,
            expiresIn=86400,
            by_sms=True,
        )

    def _fetch_positions_sync(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            import robin_stocks.robinhood as robinhood  # type: ignore[import]
        except ImportError:
            return []
        try:
            self._login(robinhood)
            raw = robinhood.account.build_holdings()
            positions = []
            for ticker, payload in raw.items():
                shares = float(payload.get("quantity", 0) or 0)
                avg_cost = float(payload.get("average_buy_price", 0) or 0)
                if shares <= 0:
                    continue
                positions.append({
                    "ticker": ticker.upper(),
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "thesis": "Synced from Robinhood",
                })
            return positions
        except Exception:
            return []

    def _sync_sync(self) -> tuple[list[dict], RobinhoodSyncResult]:
        if not self.enabled:
            return [], RobinhoodSyncResult(ok=False, error="Robinhood credentials not configured")
        try:
            import robin_stocks.robinhood as robinhood  # type: ignore[import]
        except ImportError:
            return [], RobinhoodSyncResult(
                ok=False, error="robin_stocks not installed. Run: pip install robin_stocks"
            )
        try:
            self._login(robinhood)
            raw = robinhood.account.build_holdings()
            positions = []
            for ticker, payload in raw.items():
                shares = float(payload.get("quantity", 0) or 0)
                avg_cost = float(payload.get("average_buy_price", 0) or 0)
                if shares <= 0:
                    continue
                positions.append({
                    "ticker": ticker.upper(),
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "thesis": "Synced from Robinhood",
                })
            return positions, RobinhoodSyncResult(ok=True, positions_synced=len(positions))
        except Exception as exc:
            return [], RobinhoodSyncResult(ok=False, error=str(exc))


class PlaidConnector:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self._last_sync: PlaidSyncResult | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.plaid_client_id and self.config.plaid_secret and self.config.plaid_access_token)

    @property
    def last_sync(self) -> PlaidSyncResult | None:
        return self._last_sync

    def _base_payload(self) -> dict:
        return {
            "client_id": self.config.plaid_client_id,
            "secret": self.config.plaid_secret,
            "access_token": self.config.plaid_access_token,
        }

    def _url(self, path: str) -> str:
        return f"{self.config.plaid_base_url.rstrip('/')}{path}"

    async def fetch_expenses(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            return await self._fetch_transactions_range(days=30)
        except Exception:
            return []

    async def _fetch_transactions_range(self, days: int = 30) -> list[dict]:
        end_dt = date.today()
        start_dt = date.fromordinal(end_dt.toordinal() - days)
        payload = {
            **self._base_payload(),
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "options": {"count": 500, "offset": 0, "include_personal_finance_category": True},
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(self._url("/transactions/get"), json=payload)
            resp.raise_for_status()
            data = resp.json()
        return self._normalize_transactions(data.get("transactions", []))

    async def sync(self, days: int = 90) -> PlaidSyncResult:
        if not self.enabled:
            result = PlaidSyncResult(ok=False, error="Plaid credentials not configured")
            self._last_sync = result
            return result
        try:
            transactions, accounts = await asyncio.gather(
                self._fetch_transactions_range(days=days),
                self._fetch_accounts(),
                return_exceptions=True,
            )
            tx_list: list[dict] = transactions if isinstance(transactions, list) else []
            acc_list: list[dict] = accounts if isinstance(accounts, list) else []
            errors = " | ".join(str(x) for x in [transactions, accounts] if isinstance(x, Exception))
            result = PlaidSyncResult(
                ok=not errors,
                transactions_synced=len(tx_list),
                accounts_found=len(acc_list),
                error=errors,
            )
            self._last_sync = result
            return result
        except Exception as exc:
            result = PlaidSyncResult(ok=False, error=str(exc))
            self._last_sync = result
            return result

    async def _fetch_accounts(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(self._url("/accounts/balance/get"), json=self._base_payload())
            resp.raise_for_status()
            return resp.json().get("accounts", [])

    async def fetch_account_balances(self) -> list[dict]:
        if not self.enabled:
            return []
        try:
            accounts = await self._fetch_accounts()
            return [
                {
                    "name": acc.get("name", "Account"),
                    "type": acc.get("type", ""),
                    "subtype": acc.get("subtype", ""),
                    "current_balance": (acc.get("balances") or {}).get("current") or 0.0,
                    "available_balance": (acc.get("balances") or {}).get("available"),
                    "currency": (acc.get("balances") or {}).get("iso_currency_code", "USD"),
                }
                for acc in accounts
            ]
        except Exception:
            return []

    @staticmethod
    def _normalize_transactions(raw: list[dict]) -> list[dict]:
        result = []
        for tx in raw:
            amount = float(tx.get("amount", 0))
            if amount <= 0:
                continue
            pfc = tx.get("personal_finance_category") or {}
            if pfc:
                category = pfc.get("primary") or pfc.get("detailed") or "other"
            else:
                cats = tx.get("category") or []
                category = cats[-1] if cats else "other"
            result.append({
                "category": category.lower().replace("_", " "),
                "amount": amount,
                "merchant": tx.get("merchant_name") or tx.get("name") or "Unknown",
                "incurred_on": tx["date"],
                "pending": tx.get("pending", False),
            })
        return result


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
            if payload.get("amount", 0) > 0 and not payload.get("pending", False)
        ]
        all_expenses = sorted(expenses + external_expenses, key=lambda item: item.incurred_on, reverse=True)
        expenses_month = sum(item.amount for item in all_expenses)

        # Main account cash = starting cash - all expenses (paper trading is independent).
        main_account_cash = max(0.0, profile.cash_balance - expenses_month)
        # Recompute total_value after deducting expenses from cash
        total_value = total_value - profile.cash_balance + main_account_cash

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
            cash=main_account_cash,
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
    """Cash-aware paper trading: enforces buying power, tracks realized + unrealized PnL."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def _compute_cash_state(self, user_id: str, starting_cash: float):
        """Walk trade history and compute current cash, deployed capital, realized PnL,
        FIFO open lots, and a list of closed-trade pairs with realized P&L."""
        trades = self.store.list_sim_trades(user_id)
        cash = starting_cash
        realized_pnl = 0.0
        open_lots: dict[str, list[list]] = defaultdict(list)
        net_shares: dict[str, float] = defaultdict(float)
        cost_basis: dict[str, float] = defaultdict(float)
        closed_trades: list[ClosedTrade] = []

        for trade in trades:
            side = trade.side.upper()
            if side == "BUY":
                cash -= trade.shares * trade.price
                open_lots[trade.ticker].append([trade.shares, trade.price])
                net_shares[trade.ticker] += trade.shares
                cost_basis[trade.ticker] += trade.shares * trade.price
            elif side == "SELL":
                cash += trade.shares * trade.price
                shares_to_close = trade.shares
                lots = open_lots[trade.ticker]
                while shares_to_close > 1e-9 and lots:
                    lot_shares, lot_price = lots[0]
                    take = min(lot_shares, shares_to_close)
                    pnl = take * (trade.price - lot_price)
                    pnl_pct = ((trade.price / lot_price) - 1) * 100 if lot_price else 0.0
                    realized_pnl += pnl
                    closed_trades.append(
                        ClosedTrade(
                            ticker=trade.ticker,
                            shares=take,
                            entry_price=lot_price,
                            exit_price=trade.price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            closed_at=trade.created_at,
                        )
                    )
                    lots[0][0] -= take
                    shares_to_close -= take
                    if lots[0][0] <= 1e-9:
                        lots.pop(0)
                net_shares[trade.ticker] -= trade.shares
                cost_basis[trade.ticker] -= trade.shares * (
                    cost_basis[trade.ticker] / max(net_shares[trade.ticker] + trade.shares, 1e-9)
                )

        deployed = sum(
            sum(shares * price for shares, price in lots)
            for lots in open_lots.values()
        )
        return cash, deployed, realized_pnl, dict(open_lots), closed_trades

    def execute_trade(
        self,
        user_id: str,
        ticker: str,
        side: str,
        shares: float,
        price: float,
        starting_cash: float = 10000.0,
    ) -> tuple[bool, str]:
        """Returns (ok, message). Enforces cash limits on BUY and share availability on SELL."""
        side = side.upper()
        if shares <= 0:
            return False, "Share count must be positive"
        if price <= 0:
            return False, "Invalid price"

        cash, deployed, _, open_lots, _ = self._compute_cash_state(user_id, starting_cash)

        if side == "BUY":
            cost = shares * price
            if cost > cash + 0.01:
                return False, f"Insufficient cash: need ${cost:,.2f}, have ${cash:,.2f}"
        elif side == "SELL":
            held = sum(s for s, _ in open_lots.get(ticker, []))
            if shares > held + 1e-6:
                return False, f"Cannot sell {shares} {ticker}: only hold {held:.2f}"
        else:
            return False, f"Unknown side: {side}"

        self.store.add_sim_trade(user_id, ticker, side, shares, price)
        return True, f"{side} {shares:.2f} {ticker} @ ${price:.2f}"

    def build_book(
        self,
        user_id: str,
        snapshots: dict[str, QuoteSnapshot],
        starting_cash: float = 10000.0,
    ) -> SimulationBook:
        cash, deployed, realized_pnl, open_lots, closed_trades = self._compute_cash_state(user_id, starting_cash)
        all_trades = self.store.list_sim_trades(user_id)

        positions: list[SimulationPosition] = []
        unrealized_pnl = 0.0
        gross_exposure = 0.0
        for ticker, lots in open_lots.items():
            total_shares = sum(s for s, _ in lots)
            if abs(total_shares) < 1e-9:
                continue
            cost_basis = sum(s * p for s, p in lots)
            avg_entry = cost_basis / total_shares
            snapshot = snapshots.get(ticker)
            market_price = snapshot.last if snapshot else avg_entry
            exposure = total_shares * market_price
            pnl = (market_price - avg_entry) * total_shares
            unrealized_pnl += pnl
            gross_exposure += abs(exposure)
            positions.append(
                SimulationPosition(
                    ticker=ticker,
                    net_shares=total_shares,
                    avg_entry=avg_entry,
                    market_price=market_price,
                    exposure=exposure,
                    pnl_unrealized=pnl,
                )
            )

        total_pnl = realized_pnl + unrealized_pnl
        total_value = cash + gross_exposure
        return_pct = (total_pnl / starting_cash * 100) if starting_cash > 0 else 0.0

        return SimulationBook(
            equity=unrealized_pnl,
            gross_exposure=gross_exposure,
            starting_cash=starting_cash,
            cash_available=cash,
            cash_deployed=deployed,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            total_value=total_value,
            return_pct=return_pct,
            positions=sorted(positions, key=lambda p: abs(p.exposure), reverse=True),
            trades=all_trades[-20:],
            closed_trades=closed_trades[-50:],
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
                        timestamp=datetime.now(timezone.utc),
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
                        timestamp=datetime.now(timezone.utc),
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
    """Gemini 2.0 Flash powered chat — falls back to local logic if no API key."""

    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

    def __init__(self, store: SQLiteStore, settings=None) -> None:
        self.store = store
        self.settings = settings

    @property
    def _gemini_enabled(self) -> bool:
        return bool(self.settings and self.settings.gemini_api_key)

    def _build_system_prompt(self, profile, brain, portfolio, snapshots, news_map) -> str:
        positions_text = "".join(
            f"  - {p.ticker}: {p.shares:.2f}sh @ ${p.avg_cost:.2f}, now ${p.market_price:.2f}, "
            f"PnL ${p.pnl_unrealized:+.2f}, alloc {p.allocation:.0%}\n"
            for p in portfolio.positions[:8]
        )
        market_text = "".join(
            f"  - {t}: ${s.last:.2f} ({s.change_percent:+.2f}%), trend={s.technicals.trend}, "
            f"RSI={s.technicals.rsi:.1f}, prob_up={s.prediction.probability_up:.0%}\n"
            for t, s in list(snapshots.items())[:8]
        )
        news_text = "".join(
            f"  - [{t}] {item.title} (sentiment={item.sentiment})\n"
            for t, items in list(news_map.items())[:4]
            for item in items[:2]
        )
        return f"""You are Neon FinBrain, an elite AI financial copilot embedded in a real-time trading dashboard.
You are sharp, direct, and data-driven. Give specific actionable advice — never generic disclaimers.
You have full context: live market data, ML predictions, portfolio positions, and news sentiment.

USER PROFILE:
- Horizon: {profile.investment_horizon} | Risk: {profile.risk_tolerance}
- Budget: ${profile.monthly_budget:,.0f}/mo | Cash: ${profile.cash_balance:,.0f}
- Watchlist: {", ".join(profile.watchlist)}

PORTFOLIO (${portfolio.total_value:,.2f} total, day PnL ${portfolio.day_pnl:+,.2f}, risk {portfolio.risk_score:.1f}/100):
{positions_text}
LIVE MARKET:
{market_text}
AI BRAIN (regime={brain.market_regime}, posture={brain.risk_posture}, confidence={brain.confidence:.0%}):
- Summary: {brain.summary}
- Day-trade picks: {", ".join(brain.top_day_trade_tickers or ["none"])}
- Long-term picks: {", ".join(brain.top_long_term_tickers or ["none"])}
- Notes: {"; ".join(brain.strategy_notes or [])}

NEWS:
{news_text}
Style rules — STRICTLY ENFORCED:
- Be conversational and varied. Never repeat the same phrasing or opening twice in a row.
- Reference specific tickers, prices, RSI levels, probability numbers from the data above.
- Keep responses 2–4 sentences unless explicitly asked for detail.
- Each response should feel fresh — different sentence structure, different angle, different emphasis.
- Never say you lack real-time data — you have everything above.
- Never give generic disclaimers like "consult a financial advisor". Be the advisor.
- If you already answered something similar in chat history, take a NEW angle or go deeper, don't restate."""

    async def _call_gemini(self, user_id, message, profile, brain, portfolio, snapshots, news_map) -> str:
        """Call Gemini with full visibility into failures — never silently returns empty."""
        import sys
        system_prompt = self._build_system_prompt(profile, brain, portfolio, snapshots, news_map)
        history = self.store.list_recent_chat_messages(user_id, limit=10)
        contents = [
            {"role": "user", "parts": [{"text": f"[SYSTEM CONTEXT]\n{system_prompt}"}]},
            {"role": "model", "parts": [{"text": "Understood. Full live context loaded. Ready to advise."}]},
        ]
        # history already contains the latest "user" message we just stored, so skip it
        for turn in history[:-1]:
            role = "model" if turn.role == "assistant" else "user"
            txt  = (turn.message or "").strip()
            if txt:
                contents.append({"role": role, "parts": [{"text": txt}]})
        contents.append({"role": "user", "parts": [{"text": message}]})

        url = self.GEMINI_URL.format(model=self.settings.gemini_model)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.85,
                "maxOutputTokens": 800,
                "topP": 0.95,
                "topK": 40,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        print(f"[Gemini] → POST {self.settings.gemini_model}, history turns={len(contents)}", file=sys.stderr)

        # Try the configured model first, then fall back through known-good options if 404
        models_to_try = [self.settings.gemini_model]
        for fallback in ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"):
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_err = None
        resp = None
        async with httpx.AsyncClient(timeout=30) as client:
            for model_name in models_to_try:
                attempt_url = self.GEMINI_URL.format(model=model_name)
                resp = await client.post(attempt_url, params={"key": self.settings.gemini_api_key}, json=payload)
                if resp.status_code == 200:
                    if model_name != self.settings.gemini_model:
                        print(f"[Gemini] ⚠ '{self.settings.gemini_model}' failed, "
                              f"successfully used '{model_name}' instead. "
                              f"Update GEMINI_MODEL in .env.", file=sys.stderr)
                    break
                if resp.status_code == 404:
                    last_err = f"404 model not found: {model_name}"
                    print(f"[Gemini] {last_err}, trying next…", file=sys.stderr)
                    continue
                # non-404 error: stop and report
                break

        if resp is None or resp.status_code != 200:
            err_body = resp.text[:500] if resp else (last_err or "no response")
            print(f"[Gemini] HTTP {resp.status_code if resp else 'N/A'}: {err_body}", file=sys.stderr)
            raise RuntimeError(f"Gemini API returned {resp.status_code if resp else 'error'}: {err_body[:200]}")

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            block = data.get("promptFeedback", {}).get("blockReason", "no candidates returned")
            print(f"[Gemini] blocked: {block} | full: {data}", file=sys.stderr)
            raise RuntimeError(f"Gemini returned no candidates ({block})")

        cand = candidates[0]
        finish = cand.get("finishReason", "")
        parts = cand.get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()

        if not text:
            print(f"[Gemini] empty text, finish={finish}, raw={cand}", file=sys.stderr)
            raise RuntimeError(f"Gemini returned empty response (finish={finish})")

        print(f"[Gemini] ← {len(text)} chars, finish={finish}", file=sys.stderr)
        return text

    async def handle_message(self, user_id, message, profile, brain, portfolio, snapshots, news_map):
        self.store.add_chat_message(user_id, "user", message)
        updated_profile = self._update_profile_from_message(profile, message)
        profile_changed = updated_profile != profile
        active_profile = updated_profile if profile_changed else profile
        if profile_changed:
            self.store.save_user_profile(updated_profile)
        if any(w in message.lower() for w in ("goal", "today", "budget", "focus", "need")):
            self.store.add_daily_goal(user_id, message)
        tickers = self._extract_tickers(message, active_profile.watchlist)

        response = None
        gemini_error = None
        if self._gemini_enabled:
            try:
                response = await self._call_gemini(
                    user_id, message, active_profile, brain, portfolio, snapshots, news_map
                )
            except Exception as exc:
                import traceback, sys
                gemini_error = str(exc)
                print(f"[Gemini error] {exc}", file=sys.stderr)
                traceback.print_exc()
                response = None
        if not response:
            local = self._local_response(
                message, active_profile, brain, portfolio, snapshots, news_map, tickers
            )
            if gemini_error:
                response = f"⚠ Gemini error ({gemini_error[:120]}). Local fallback:\n\n{local}"
            else:
                response = local

        self.store.add_chat_message(user_id, "assistant", response)
        return ChatReply(
            response=response,
            suggestions=self.suggested_prompts(),
            referenced_tickers=tickers,
            updated_profile=updated_profile if profile_changed else None,
        )

    @staticmethod
    def suggested_prompts():
        return [
            "What are the best day-trade setups right now?",
            "Review my portfolio risk and suggest adjustments.",
            "Which long-term names fit my current goal profile?",
            "Explain why the top ticker is being recommended.",
            "What is the market regime telling us today?",
            "Should I reduce risk given my current expenses?",
        ]

    @staticmethod
    def _local_response(message, profile, brain, portfolio, snapshots, news_map, tickers):
        lower = message.lower()
        if "portfolio" in lower or "risk" in lower:
            return (
                f"Portfolio value ${portfolio.total_value:,.0f}, expenses ${portfolio.expenses_month:,.0f}/mo, "
                f"risk {portfolio.risk_score:.1f}/100. "
                f"{brain.strategy_notes[0] if brain.strategy_notes else 'Hold risk steady until signals improve.'}"
            )
        if "day trade" in lower or "scalp" in lower:
            return (
                f"Best day-trade candidates: {', '.join(brain.top_day_trade_tickers or ['No clear setup'])}. "
                f"Regime is {brain.market_regime}."
            )
        if "long term" in lower or "invest" in lower:
            return f"Best long-term picks: {', '.join(brain.top_long_term_tickers or ['No strong setup'])}."
        if tickers:
            pieces = []
            for t in tickers[:3]:
                snap = snapshots.get(t)
                if not snap:
                    continue
                items = news_map.get(t, [])
                headline = items[0].title if items else "no major headline"
                pieces.append(
                    f"{t} at ${snap.last:,.2f}, trend={snap.technicals.trend}, "
                    f"prob_up={snap.prediction.probability_up:.0%}. Catalyst: {headline}."
                )
            return " ".join(pieces) or brain.summary
        return (
            f"{brain.summary} Day-trade: {', '.join(brain.top_day_trade_tickers or ['none'])}. "
            f"Long-term: {', '.join(brain.top_long_term_tickers or ['none'])}."
        )

    def _update_profile_from_message(self, profile, message):
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
        m = re.search(r"\$?(\d{3,6})", message.replace(",", ""))
        if m and "budget" in lower:
            updated.monthly_budget = float(m.group(1))
        return updated

    def _extract_tickers(self, message, watchlist):
        mentioned = {m.group(0).upper() for m in self.TICKER_PATTERN.finditer(message)}
        return [t for t in watchlist if t in mentioned]



@dataclass(slots=True)
class DashboardState:
    profile: UserProfile
    market: dict[str, QuoteSnapshot] = field(default_factory=dict)
    news: dict[str, list[NewsItem]] = field(default_factory=dict)
    portfolio: PortfolioSummary = field(default_factory=PortfolioSummary)
    brain: BrainDecision = field(default_factory=BrainDecision)
    simulation: SimulationBook = field(default_factory=SimulationBook)
    scenario: ScenarioResult | None = None
    refreshed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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
        self.chat_service = ChatService(self.store, self.config)
        self._lock = asyncio.Lock()
        self._bootstrapped = False
        self.active_user_id = config.default_user_id
        self.state: DashboardState | None = None

    def set_active_user(self, user_id: str) -> None:
        if user_id and user_id != self.active_user_id:
            self.active_user_id = user_id
            self._bootstrapped = False
            self.state = None

    async def ensure_bootstrapped(self) -> None:
        if self._bootstrapped:
            return
        self.store.initialize()
        profile = self.store.load_user_profile(self.active_user_id, self.config.watchlist)
        self.state = DashboardState(profile=profile)
        self._bootstrapped = True
        await self.refresh_all(force=True)

    async def ensure_fresh(self) -> None:
        await self.ensure_bootstrapped()
        assert self.state is not None
        age = (datetime.now(timezone.utc) - self.state.refreshed_at).total_seconds()
        if age >= self.config.refresh_seconds:
            await self.refresh_all()

    async def refresh_all(self, force: bool = False) -> DashboardState:
        await self.ensure_bootstrapped()
        async with self._lock:
            profile = self.store.load_user_profile(self.active_user_id, self.config.watchlist)
            market, news = await asyncio.gather(
                self.market_service.get_snapshots(profile.watchlist, force=force),
                self.news_service.get_news_for_tickers(profile.watchlist[:6], force=force),
            )
            enriched_news = {ticker: self.sentiment_engine.enrich(items) for ticker, items in news.items()}
            portfolio = await self.portfolio_service.build_summary(profile.user_id, profile, market)
            simulation = self.simulator.build_book(profile.user_id, market, starting_cash=10000.0)
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
                refreshed_at=datetime.now(timezone.utc),
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
        profile = self.store.load_user_profile(self.active_user_id, self.config.watchlist)
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
        self.store.upsert_position(self.active_user_id, ticker, shares, avg_cost, thesis)
        return await self.refresh_all(force=True)

    async def add_expense(self, category: str, amount: float, merchant: str) -> DashboardState:
        await self.ensure_bootstrapped()
        self.store.add_expense(self.active_user_id, category, amount, merchant)
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
        ok, msg = self.simulator.execute_trade(
            self.active_user_id, ticker, side, shares, snapshot.last,
            starting_cash=10000.0,
        )
        if not ok:
            raise ValueError(msg)
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
        reply = await self.chat_service.handle_message(
            self.active_user_id,
            message,
            state.profile,
            state.brain,
            state.portfolio,
            state.market,
            state.news,
        )
        refreshed = await self.refresh_all()
        return reply, refreshed

    async def sync_robinhood(self) -> RobinhoodSyncResult:
        await self.ensure_bootstrapped()
        result = await self.portfolio_service.robinhood.sync()
        if result.ok and result.positions_synced > 0:
            positions, _ = await asyncio.to_thread(self.portfolio_service.robinhood._sync_sync)
            for pos in positions:
                self.store.upsert_position(
                    self.active_user_id,
                    pos["ticker"],
                    pos["shares"],
                    pos["avg_cost"],
                    pos.get("thesis", "Synced from Robinhood"),
                )
            await self.refresh_all(force=True)
        return result

    async def sync_plaid(self, days: int = 90) -> PlaidSyncResult:
        await self.ensure_bootstrapped()
        result = await self.portfolio_service.plaid.sync(days=days)
        if result.ok and result.transactions_synced > 0:
            try:
                transactions = await self.portfolio_service.plaid._fetch_transactions_range(days=days)
                for tx in transactions:
                    if tx.get("amount", 0) > 0 and not tx.get("pending", False):
                        self.store.add_expense(
                            self.active_user_id,
                            tx["category"],
                            tx["amount"],
                            tx.get("merchant", "Plaid"),
                        )
            except Exception:
                pass
            await self.refresh_all(force=True)
        return result

    def connector_status(self) -> dict:
        rh = self.portfolio_service.robinhood
        pl = self.portfolio_service.plaid
        rh_last = rh.last_sync
        pl_last = pl.last_sync
        return {
            "robinhood": {
                "configured": rh.enabled,
                "last_sync": rh_last.synced_at if rh_last else None,
                "last_sync_ok": rh_last.ok if rh_last else None,
                "positions_synced": rh_last.positions_synced if rh_last else None,
                "error": rh_last.error if rh_last and not rh_last.ok else None,
            },
            "plaid": {
                "configured": pl.enabled,
                "last_sync": pl_last.synced_at if pl_last else None,
                "last_sync_ok": pl_last.ok if pl_last else None,
                "transactions_synced": pl_last.transactions_synced if pl_last else None,
                "accounts_found": pl_last.accounts_found if pl_last else None,
                "error": pl_last.error if pl_last and not pl_last.ok else None,
            },
        }


platform = PlatformOrchestrator(settings)


def current_auth_user() -> dict[str, Any] | None:
    email = app.storage.user.get("user_id")
    if not email:
        return None
    return platform.store.get_user_by_email(str(email))


def login_session(user: dict[str, Any]) -> None:
    app.storage.user["authenticated"] = True
    app.storage.user["user_id"] = user["user_id"]
    app.storage.user["full_name"] = user["full_name"]
    platform.set_active_user(user["user_id"])


def logout_session() -> None:
    app.storage.user.clear()
    platform.set_active_user(settings.default_user_id)


def require_auth() -> bool:
    if not app.storage.user.get("authenticated"):
        ui.navigate.to("/login")
        return False
    platform.set_active_user(str(app.storage.user.get("user_id")))
    return True


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
          width: min(440px, calc(100vw - 1rem));
          max-height: min(720px, calc(100vh - 2rem));
          padding: 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .chat-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 0.75rem;
        }
        .chat-scroll {
          max-height: min(390px, calc(100vh - 320px));
          overflow-y: auto;
          padding-right: 0.25rem;
        }
        .chat-input-box textarea {
          min-height: 78px;
          resize: vertical;
        }
        .chat-fab {
          border: 1px solid rgba(76, 243, 255, 0.45);
          background: linear-gradient(135deg, rgba(76, 243, 255, 0.94), rgba(255, 93, 211, 0.88));
          color: #040612;
          box-shadow: 0 18px 46px rgba(76, 243, 255, 0.22);
          border-radius: 999px;
          padding: 0.85rem 1.05rem;
          font-weight: 800;
          letter-spacing: 0.02em;
        }
        .typing { animation: pulse 1.1s ease-in-out infinite; }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        @media (max-width: 700px) {
          .chat-card {
            left: 0.5rem !important;
            right: 0.5rem !important;
            bottom: 0.5rem !important;
            width: calc(100vw - 1rem);
          }
          .chat-fab {
            right: 0.75rem !important;
            bottom: 0.75rem !important;
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


@app.get("/api/connectors/status")
async def api_connector_status():
    await platform.ensure_bootstrapped()
    return jsonable_encoder(platform.connector_status())


@app.post("/api/connectors/robinhood/sync")
async def api_robinhood_sync():
    result = await platform.sync_robinhood()
    return jsonable_encoder({
        "ok": result.ok,
        "positions_synced": result.positions_synced,
        "error": result.error,
        "synced_at": result.synced_at,
    })


@app.post("/api/connectors/plaid/sync")
async def api_plaid_sync(payload: dict | None = None):
    days = int((payload or {}).get("days", 90))
    result = await platform.sync_plaid(days=days)
    return jsonable_encoder({
        "ok": result.ok,
        "transactions_synced": result.transactions_synced,
        "accounts_found": result.accounts_found,
        "error": result.error,
        "synced_at": result.synced_at,
    })


@app.get("/api/connectors/plaid/accounts")
async def api_plaid_accounts():
    await platform.ensure_bootstrapped()
    accounts = await platform.portfolio_service.plaid.fetch_account_balances()
    return jsonable_encoder({"accounts": accounts})


@ui.page("/signup")
async def signup_page() -> None:
    inject_theme()
    ui.dark_mode().enable()
    platform.store.initialize()
    if app.storage.user.get("authenticated"):
        ui.navigate.to("/")
        return

    async def handle_signup() -> None:
        if password.value != confirm_password.value:
            ui.notify("Passwords do not match.", color="negative")
            return
        ok, message = platform.store.create_user(full_name.value or "", email.value or "", password.value or "")
        if not ok:
            ui.notify(message, color="negative")
            return
        user = platform.store.authenticate_user(email.value or "", password.value or "")
        if user:
            login_session(user)
            ui.notify(message, color="positive")
            ui.navigate.to("/")

    with ui.column().classes("page-shell items-center justify-center min-h-screen"):
        with ui.card().classes("glass-card auth-card w-full max-w-md"):
            ui.label("Create your FinBrain account").classes("title-font text-3xl")
            ui.label("Your portfolio, goals, chat history, expenses, and paper trades will be saved only under your login.").classes("subtle")
            full_name = ui.input("Full name").classes("w-full")
            email = ui.input("Email").props("type=email").classes("w-full")
            password = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")
            confirm_password = ui.input("Confirm password", password=True, password_toggle_button=True).classes("w-full")
            ui.button("Create Account", on_click=handle_signup).classes("w-full")
            with ui.row().classes("w-full justify-center"):
                ui.label("Already have an account?").classes("subtle")
                ui.link("Login", "/login").classes("text-cyan-300")


@ui.page("/login")
async def login_page() -> None:
    inject_theme()
    ui.dark_mode().enable()
    platform.store.initialize()
    if app.storage.user.get("authenticated"):
        ui.navigate.to("/")
        return

    async def handle_login() -> None:
        user = platform.store.authenticate_user(email.value or "", password.value or "")
        if not user:
            ui.notify("Invalid email or password.", color="negative")
            return
        login_session(user)
        ui.notify("Logged in successfully.", color="positive")
        ui.navigate.to("/")

    with ui.column().classes("page-shell items-center justify-center min-h-screen"):
        with ui.card().classes("glass-card auth-card w-full max-w-md"):
            ui.label("Login to Alfa Stock AI").classes("title-font text-3xl")
            ui.label("Use your own account to access your protected AI finance dashboard.").classes("subtle")
            email = ui.input("Email").props("type=email").classes("w-full")
            password = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")
            ui.button("Login", on_click=handle_login).classes("w-full")
            with ui.row().classes("w-full justify-center"):
                ui.label("Need an account?").classes("subtle")
                ui.link("Sign up", "/signup").classes("text-cyan-300")


@ui.page("/logout")
async def logout_page() -> None:
    logout_session()
    ui.notify("You have been logged out.", color="positive")
    ui.navigate.to("/login")


@ui.page("/")
async def home() -> None:
    inject_theme()
    ui.dark_mode().enable()
    if not require_auth():
        return
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
        ticker_val = (position_ticker.value or "").strip().upper()
        shares_val = float(position_shares.value or 0)
        cost_val   = float(position_cost.value or 0)
        if not ticker_val or shares_val <= 0 or cost_val <= 0:
            ui.notify("⚠ Enter ticker, shares > 0, and avg cost > 0", type="warning")
            return
        try:
            await platform.add_position(ticker_val, shares_val, cost_val, position_thesis.value or "")
            refresh_all_sections()
            ui.notify(f"✅ Added {shares_val:.2f} {ticker_val} @ {money(cost_val)}", type="positive")
            position_ticker.set_value("")
            position_shares.set_value(0)
            position_cost.set_value(0)
            position_thesis.set_value("")
        except Exception as exc:
            ui.notify(f"❌ {exc}", type="negative", timeout=6000)

    async def save_expense() -> None:
        category_val = (expense_category.value or "").strip()
        amount_val   = float(expense_amount.value or 0)
        if not category_val or amount_val <= 0:
            ui.notify("⚠ Enter category and amount > 0", type="warning")
            return
        try:
            await platform.add_expense(category_val, amount_val, expense_merchant.value or "")
            refresh_all_sections()
            ui.notify(f"✅ Logged ${amount_val:.2f} for {category_val} (deducted from main account)", type="positive")
            expense_category.set_value("")
            expense_amount.set_value(0)
            expense_merchant.set_value("")
        except Exception as exc:
            ui.notify(f"❌ {exc}", type="negative", timeout=6000)

    async def execute_trade(side: str) -> None:
        try:
            await platform.execute_sim_trade(sim_ticker.value.upper(), side, float(sim_shares.value or 0))
            refresh_all_sections()
            ui.notify(f"✅ Paper trade: {side} {sim_shares.value} {sim_ticker.value.upper()}", type="positive")
        except ValueError as exc:
            ui.notify(f"❌ {exc}", type="negative", timeout=6000)
        except Exception as exc:
            ui.notify(f"❌ Trade failed: {exc}", type="negative", timeout=6000)

    async def update_scenario() -> None:
        await platform.run_scenario(float(scenario_slider.value))
        simulator_panel.refresh()

    async def send_chat(prompt: str | None = None) -> None:
        message = (prompt or chat_input.value or "").strip()
        if not message:
            ui.notify("Type a message first", type="warning")
            return

        # Optimistically clear input + show typing immediately
        chat_input.set_value("")
        typing_note.set_text("🤖 Gemini is thinking…")
        typing_note.set_visibility(True)
        chat_history.refresh()  # show the user message right away

        try:
            reply, _ = await platform.chat(message)
        except Exception as exc:
            typing_note.set_visibility(False)
            ui.notify(f"❌ Chat failed: {exc}", type="negative", timeout=8000)
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            return

        typing_note.set_visibility(False)
        refresh_all_sections()

        # Diagnostic: show whether Gemini or local fallback was used
        if platform.chat_service._gemini_enabled:
            ui.notify("✅ Gemini response received", type="positive", timeout=2000)
        else:
            ui.notify("⚠ Using local fallback (no GEMINI_API_KEY)", type="warning", timeout=4000)

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
                ui.label("Alfa Stock AI").classes("title-font text-4xl")
                ui.label(
                    "A real-time AI market brain with no-key default data feeds, local ML, news credibility scoring, paper trading, and portfolio reasoning."
                ).classes("subtle max-w-3xl")
            with ui.column().classes("items-end gap-2"):
                ui.label("No-key mode: Yahoo Finance + free RSS/news").classes("neon-pill")
                ui.label(f"Logged in as {app.storage.user.get('full_name', 'User')}").classes("neon-pill")
                ui.link("Logout", "/logout").classes("text-cyan-300")
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
                    ("Last Refresh", local_time_text(current.refreshed_at), "metric-card"),
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
            starting = current.profile.cash_balance
            cash_now = current.portfolio.cash
            spent    = current.portfolio.expenses_month
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Portfolio Tracker — Main Account").classes("title-font text-xl")

                    # Account summary row
                    with ui.row().classes("w-full gap-2 mt-2 flex-wrap"):
                        with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                            ui.label("Starting Balance").classes("subtle text-xs")
                            ui.label(money(starting)).classes("font-semibold")
                        with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                            ui.label("Total Expenses").classes("subtle text-xs")
                            ui.label(signed_money(-spent)).classes(f"font-semibold {color_class(-spent)}")
                            ui.label("this month").classes("subtle text-xs")
                        with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                            ui.label("Available Cash").classes("subtle text-xs")
                            ui.label(money(cash_now)).classes(f"font-semibold {color_class(cash_now - starting)}")
                            ui.label("after expenses").classes("subtle text-xs")
                        with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                            ui.label("Total Net Worth").classes("subtle text-xs")
                            ui.label(money(current.portfolio.total_value)).classes("font-semibold")
                            ui.label("cash + holdings").classes("subtle text-xs")

                    ui.separator()
                    ui.label("Holdings (manual entries + Robinhood)").classes("font-semibold")
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
            sim = current.simulation
            ret_color  = "positive" if sim.return_pct >= 0 else "negative"
            pnl_color  = "positive" if sim.total_pnl >= 0 else "negative"
            cash_used_pct = (sim.cash_deployed / sim.starting_cash * 100) if sim.starting_cash else 0

            # ── SECTION 1: Account Summary (Live) ─────────────────────────
            with ui.card().classes("glass-card w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Paper Trading Account").classes("title-font text-xl")
                    ui.label("● LIVE").classes("subtle text-xs positive")
                ui.label(
                    f"Starting capital ${sim.starting_cash:,.0f}  →  Total value {money(sim.total_value)}  "
                    f"({sim.return_pct:+.2f}%)"
                ).classes("subtle")

                with ui.row().classes("w-full gap-2 mt-2 flex-wrap"):
                    with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                        ui.label("Cash Available").classes("subtle text-xs")
                        ui.label(money(sim.cash_available)).classes("font-semibold")
                        ui.label(f"{(sim.cash_available/sim.starting_cash*100 if sim.starting_cash else 0):.1f}% of capital").classes("subtle text-xs")
                    with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                        ui.label("Holdings Value").classes("subtle text-xs")
                        ui.label(money(sim.gross_exposure)).classes("font-semibold")
                        ui.label(f"{cash_used_pct:.1f}% utilized").classes("subtle text-xs")
                    with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                        ui.label("Realized P&L").classes("subtle text-xs")
                        ui.label(signed_money(sim.realized_pnl)).classes(f"font-semibold {color_class(sim.realized_pnl)}")
                        ui.label("from closed trades").classes("subtle text-xs")
                    with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                        ui.label("Unrealized P&L").classes("subtle text-xs")
                        ui.label(signed_money(sim.equity)).classes(f"font-semibold {color_class(sim.equity)}")
                        ui.label("open positions").classes("subtle text-xs")
                    with ui.column().classes("ai-badge flex-1 min-w-[140px] gap-1"):
                        ui.label("Total Return").classes("subtle text-xs")
                        ui.label(f"{sim.return_pct:+.2f}%").classes(f"font-semibold {ret_color}")
                        ui.label(signed_money(sim.total_pnl)).classes(f"subtle text-xs {pnl_color}")

                if sim.starting_cash > 0:
                    ui.label("Buying power utilization").classes("subtle text-xs mt-2")
                    ui.linear_progress(
                        value=min(1.0, cash_used_pct / 100.0),
                        color="cyan" if cash_used_pct < 80 else "orange",
                    ).classes("w-full")

            # ── SECTION 2: Currently Holding (live prices) ────────────────
            with ui.card().classes("glass-card w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Currently Holding").classes("title-font text-xl")
                    ui.label(f"{len(sim.positions)} open position{'' if len(sim.positions)==1 else 's'}").classes("subtle text-xs")

                if sim.positions:
                    # Header row
                    with ui.row().classes("w-full items-center gap-2 px-2 py-1"):
                        ui.label("Ticker").classes("subtle text-xs flex-1 min-w-[60px]")
                        ui.label("Shares").classes("subtle text-xs flex-1 min-w-[60px] text-right")
                        ui.label("Avg Cost").classes("subtle text-xs flex-1 min-w-[80px] text-right")
                        ui.label("Live Price").classes("subtle text-xs flex-1 min-w-[80px] text-right")
                        ui.label("Market Value").classes("subtle text-xs flex-1 min-w-[100px] text-right")
                        ui.label("P&L").classes("subtle text-xs flex-1 min-w-[100px] text-right")
                        ui.label("P&L %").classes("subtle text-xs flex-1 min-w-[70px] text-right")

                    for pos in sim.positions:
                        pnl_pct = ((pos.market_price / pos.avg_entry) - 1) * 100 if pos.avg_entry else 0
                        cls = color_class(pos.pnl_unrealized)
                        with ui.row().classes("w-full items-center gap-2 ai-badge"):
                            ui.label(pos.ticker).classes("font-semibold flex-1 min-w-[60px]")
                            ui.label(f"{pos.net_shares:.2f}").classes("flex-1 min-w-[60px] text-right")
                            ui.label(money(pos.avg_entry)).classes("subtle flex-1 min-w-[80px] text-right")
                            ui.label(money(pos.market_price)).classes(f"font-semibold flex-1 min-w-[80px] text-right")
                            ui.label(money(pos.exposure)).classes("flex-1 min-w-[100px] text-right")
                            ui.label(signed_money(pos.pnl_unrealized)).classes(f"font-semibold flex-1 min-w-[100px] text-right {cls}")
                            ui.label(f"{pnl_pct:+.2f}%").classes(f"font-semibold flex-1 min-w-[70px] text-right {cls}")
                else:
                    ui.label("No open positions. When you buy, holdings appear here. When you sell, they disappear.").classes("subtle")

            # ── SECTION 3: Trade Log (closed trades + P&L) ────────────────
            with ui.card().classes("glass-card w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Trade Log").classes("title-font text-xl")
                    if sim.closed_trades:
                        wins = sum(1 for t in sim.closed_trades if t.pnl > 0)
                        losses = sum(1 for t in sim.closed_trades if t.pnl < 0)
                        win_rate = wins / max(wins + losses, 1) * 100
                        ui.label(
                            f"{len(sim.closed_trades)} closed · {wins}W / {losses}L · "
                            f"{win_rate:.0f}% win rate"
                        ).classes("subtle text-xs")
                    else:
                        ui.label("No closed trades yet").classes("subtle text-xs")

                if sim.closed_trades:
                    # Header row
                    with ui.row().classes("w-full items-center gap-2 px-2 py-1"):
                        ui.label("Closed").classes("subtle text-xs flex-1 min-w-[80px]")
                        ui.label("Ticker").classes("subtle text-xs flex-1 min-w-[60px]")
                        ui.label("Shares").classes("subtle text-xs flex-1 min-w-[60px] text-right")
                        ui.label("Entry").classes("subtle text-xs flex-1 min-w-[70px] text-right")
                        ui.label("Exit").classes("subtle text-xs flex-1 min-w-[70px] text-right")
                        ui.label("P&L").classes("subtle text-xs flex-1 min-w-[90px] text-right")
                        ui.label("P&L %").classes("subtle text-xs flex-1 min-w-[70px] text-right")

                    # Most recent first
                    for trade in sim.closed_trades[::-1][:30]:
                        cls = color_class(trade.pnl)
                        outcome = "WIN" if trade.pnl > 0 else ("LOSS" if trade.pnl < 0 else "EVEN")
                        ts = trade.closed_at.strftime("%m/%d %H:%M") if hasattr(trade.closed_at, "strftime") else str(trade.closed_at)[:16]
                        with ui.row().classes("w-full items-center gap-2 ai-badge"):
                            ui.label(ts).classes("subtle text-xs flex-1 min-w-[80px]")
                            ui.label(trade.ticker).classes("font-semibold flex-1 min-w-[60px]")
                            ui.label(f"{trade.shares:.2f}").classes("flex-1 min-w-[60px] text-right")
                            ui.label(money(trade.entry_price)).classes("subtle flex-1 min-w-[70px] text-right")
                            ui.label(money(trade.exit_price)).classes("subtle flex-1 min-w-[70px] text-right")
                            ui.label(signed_money(trade.pnl)).classes(f"font-semibold flex-1 min-w-[90px] text-right {cls}")
                            ui.label(f"{trade.pnl_pct:+.2f}%").classes(f"font-semibold flex-1 min-w-[70px] text-right {cls}")

                # Recent activity (raw trade actions)
                if sim.trades:
                    ui.separator()
                    ui.label("Recent Activity").classes("font-semibold text-sm")
                    for trade in sim.trades[-8:][::-1]:
                        side_cls = "positive" if trade.side.upper() == "BUY" else "negative"
                        ts = trade.created_at.strftime("%m/%d %H:%M") if hasattr(trade.created_at, "strftime") else str(trade.created_at)[:16]
                        with ui.row().classes("w-full items-center justify-between ai-badge"):
                            with ui.row().classes("gap-2 items-center"):
                                ui.label(ts).classes("subtle text-xs")
                                ui.label(trade.side.upper()).classes(f"font-semibold {side_cls}")
                                ui.label(f"{trade.shares:.2f} {trade.ticker}").classes("subtle")
                            ui.label(f"@ {money(trade.price)}  =  {money(trade.shares * trade.price)}").classes("subtle text-xs")

            # ── SECTION 4: What-if Scenario Engine ────────────────────────
            with ui.card().classes("glass-card w-full"):
                ui.label("What-if Scenario Engine").classes("title-font text-xl")
                ui.label(current.scenario.narrative if current.scenario else "Run a stress test").classes("subtle")
                if current.scenario:
                    for impact in current.scenario.impacts[:5]:
                        with ui.row().classes("w-full items-center justify-between ai-badge"):
                            ui.label(f"{impact.ticker} shock {signed_percent(impact.shock_pct)}")
                            ui.label(signed_money(impact.estimated_pnl)).classes(color_class(impact.estimated_pnl))
                ui.separator()
                ui.plotly(simulation_figure(current)).classes("w-full")

        hero_metrics()
        market_strip()
        chart_panel()
        ai_panel()
        portfolio_panel()
        news_panel()
        simulator_panel()


        # ── RL Trader Panel (paper trading enabled) ─────────────────────────
        with ui.card().classes("glass-card w-full"):
            ui.label("RL Trading Terminal — Paper Trading Mode").classes("title-font text-xl")
            ui.label(
                "PPO reinforcement learning agent. Backtests strategy on historical data, "
                "then generates a live BUY/SELL/HOLD signal on the most recent bar. "
                "Click Execute to place a real paper trade using your cash balance."
            ).classes("subtle mb-2")

            # Hold the trained model + signal across button clicks
            rl_state = {"model": None, "ticker": None, "win_size": None, "df": None,
                        "risk": None, "signal": None, "shares_to_trade": 0.0, "price": 0.0}

            with ui.row().classes("w-full gap-4 flex-wrap"):
                rl_ticker   = ui.input("Ticker", value="AAPL").classes("w-24")
                rl_preset   = ui.select(
                    ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
                    value="6 Months", label="Date Range"
                ).classes("w-36")
                rl_risk     = ui.select(
                    ["Conservative", "Moderate", "Aggressive"],
                    value="Moderate", label="Risk Profile"
                ).classes("w-36")
                rl_steps    = ui.number("Training Steps", value=10000, min=2000, max=50000, step=1000).classes("w-36")
                rl_window   = ui.number("Window Size", value=8, min=3, max=20, step=1).classes("w-28")
                rl_split    = ui.number("Train Split %", value=80, min=50, max=90, step=5).classes("w-28")
                rl_alloc    = ui.number("Position Size %", value=20, min=1, max=100, step=5).classes("w-32")

            rl_status      = ui.label("").classes("subtle")
            rl_metrics     = ui.html("").classes("w-full")
            rl_signal_box  = ui.html("").classes("w-full")
            rl_chart       = ui.html("").classes("w-full")
            rl_execute_btn = ui.button("📈 Execute Signal as Paper Trade").classes("self-start mt-2 hidden")

            async def run_rl_trainer() -> None:
                import pandas as _pd
                import numpy as _np

                rl_status.set_text("⏳ Fetching market data…")
                rl_metrics.set_content("")
                rl_signal_box.set_content("")
                rl_chart.set_content("")
                rl_execute_btn.classes(add="hidden")

                try:
                    from stable_baselines3 import PPO as _PPO
                    import gymnasium as _gym
                    from gymnasium import spaces as _spaces
                except ImportError:
                    rl_status.set_text("❌ Missing packages. Run: pip install stable-baselines3 gymnasium")
                    return

                ticker_sym  = rl_ticker.value.upper()
                preset_map  = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
                days_back   = preset_map.get(rl_preset.value, 180)
                end_dt      = _pd.Timestamp.today().normalize()
                start_dt    = end_dt - _pd.DateOffset(days=days_back)
                risk_choice = rl_risk.value
                n_steps_val = int(rl_steps.value or 10000)
                win_size    = int(rl_window.value or 8)
                split_pct   = int(rl_split.value or 80)

                try:
                    raw = await asyncio.to_thread(
                        lambda: yf.download(ticker_sym, start=str(start_dt.date()),
                                            end=str(end_dt.date()), auto_adjust=True, progress=False)
                    )
                    if isinstance(raw.columns, _pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                except Exception as exc:
                    rl_status.set_text(f"❌ Data fetch failed: {exc}")
                    return

                min_bars = max(win_size * 3 + 10, 30)
                if len(raw) < min_bars:
                    rl_status.set_text(f"❌ Only {len(raw)} bars. Try a wider date range.")
                    return

                def compute(df):
                    d = df.copy()
                    n = len(d)
                    w = {"rsi": max(5, min(14, n//5)), "ema_s": max(3, min(12, n//8)),
                         "ema_l": max(6, min(26, n//4)), "signal": max(3, min(9, n//10)),
                         "bb": max(5, min(20, n//5)), "vol": max(5, min(20, n//5))}
                    delta = d["Close"].diff()
                    gain  = delta.clip(lower=0).rolling(w["rsi"]).mean()
                    loss  = (-delta.clip(upper=0)).rolling(w["rsi"]).mean()
                    d["RSI"]       = 100 - 100 / (1 + gain / loss.replace(0, _np.nan))
                    ema_s          = d["Close"].ewm(span=w["ema_s"]).mean()
                    ema_l          = d["Close"].ewm(span=w["ema_l"]).mean()
                    d["MACD"]      = ema_s - ema_l
                    d["Signal"]    = d["MACD"].ewm(span=w["signal"]).mean()
                    sma            = d["Close"].rolling(w["bb"]).mean()
                    std            = d["Close"].rolling(w["bb"]).std()
                    d["BB_pct"]    = (d["Close"] - (sma - 2*std)) / (4*std + 1e-8)
                    d["DailyRet"]  = d["Close"].pct_change()
                    d["Volatility"]= d["DailyRet"].rolling(w["vol"]).std() * _np.sqrt(252) * 100
                    return d.ffill().bfill()

                df = await asyncio.to_thread(compute, raw)

                RISK_PROFILES = {
                    "Conservative": dict(loss_penalty=3.0, win_bonus=1.0, stop_loss=0.03, hold_penalty=0.0002),
                    "Moderate":     dict(loss_penalty=1.5, win_bonus=1.5, stop_loss=0.06, hold_penalty=0.0001),
                    "Aggressive":   dict(loss_penalty=0.8, win_bonus=2.5, stop_loss=0.12, hold_penalty=0.00005),
                }

                class _RLEnv(_gym.Env):
                    FEAT_COLS = ["Open","High","Low","Close","Volume","RSI","MACD","Signal","BB_pct","Volatility"]
                    def __init__(self, data, wsize, profile):
                        super().__init__()
                        self.df = data.reset_index(drop=True)
                        self.n  = len(data)
                        self.ws = wsize
                        self.p  = RISK_PROFILES[profile]
                        feats = data[self.FEAT_COLS].copy().astype(_np.float32)
                        self._mean = feats.mean(); self._std = feats.std() + 1e-8
                        self.norm = ((feats - self._mean) / self._std).values
                        self.observation_space = _spaces.Box(-10, 10, shape=(wsize * len(self.FEAT_COLS),), dtype=_np.float32)
                        self.action_space = _spaces.Discrete(3)
                        self.reset()
                    def reset(self, seed=None, options=None):
                        super().reset(seed=seed)
                        self.idx = self.ws; self.pos = 0; self.entry = 0.0
                        self.profit = 1.0; self.wins = 0; self.losses = 0
                        self.trade_log = []
                        return self._obs(), {}
                    def _obs(self):
                        return self.norm[self.idx - self.ws:self.idx].flatten().astype(_np.float32)
                    def step(self, action):
                        price = float(self.df.iloc[self.idx]["Close"])
                        reward = 0.0
                        if self.pos == 1 and (price - self.entry) / self.entry <= -self.p["stop_loss"]:
                            action = 2
                        if action == 1 and self.pos == 0:
                            self.pos = 1; self.entry = price
                        elif action == 2 and self.pos == 1:
                            pct = (price - self.entry) / self.entry
                            self.profit *= (1 + pct)
                            if pct > 0:
                                reward = pct * self.p["win_bonus"]; self.wins += 1; outcome = "WIN"
                            else:
                                reward = pct * self.p["loss_penalty"]; self.losses += 1; outcome = "LOSS"
                            self.trade_log.append({"type": outcome, "entry": round(self.entry,2),
                                                   "exit": round(price,2), "pct": round(pct*100,3)})
                            self.pos = 0
                        if self.pos == 1:
                            reward -= self.p["hold_penalty"]
                        self.idx += 1
                        done = self.idx >= self.n - 1
                        return self._obs(), reward, done, False, {
                            "total_profit": self.profit, "wins": self.wins,
                            "losses": self.losses, "trade_log": self.trade_log,
                        }

                split_idx = max(int(len(df) * split_pct / 100), win_size + 5)
                split_idx = min(split_idx, len(df) - win_size - 5)
                train_df  = df.iloc[:split_idx].copy()
                test_df   = df.iloc[split_idx:].copy()

                rl_status.set_text(f"⏳ Training PPO [{risk_choice}] for {n_steps_val:,} timesteps…")

                def _train():
                    env = _RLEnv(train_df, win_size, risk_choice)
                    ppo_n = min(512, len(train_df) - win_size - 1)
                    m = _PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=ppo_n)
                    m.learn(total_timesteps=n_steps_val)
                    return m

                try:
                    model = await asyncio.to_thread(_train)
                except Exception as exc:
                    rl_status.set_text(f"❌ Training failed: {exc}")
                    return

                # Backtest on unseen test data
                def _backtest(m):
                    env = _RLEnv(test_df, win_size, risk_choice)
                    obs, _ = env.reset()
                    while True:
                        action, _ = m.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            break
                    return info

                info       = await asyncio.to_thread(_backtest, model)
                trades     = info["trade_log"]
                agent_ret  = (info["total_profit"] - 1) * 100
                bh_ret     = (test_df["Close"].iloc[-1] / test_df["Close"].iloc[win_size] - 1) * 100
                alpha      = agent_ret - bh_ret
                wins       = info["wins"]; losses = info["losses"]; total_t = wins + losses
                win_rate   = (wins / total_t * 100) if total_t else 0
                avg_win    = _np.mean([t["pct"] for t in trades if t["type"] == "WIN"])  if any(t["type"]=="WIN"  for t in trades) else 0
                avg_loss   = _np.mean([t["pct"] for t in trades if t["type"] == "LOSS"]) if any(t["type"]=="LOSS" for t in trades) else 0

                # ─── LIVE SIGNAL on most recent bar ──────────────────────────
                live_env  = _RLEnv(df, win_size, risk_choice)
                live_env.idx = len(df) - 1
                live_obs  = live_env._obs()
                action, _ = await asyncio.to_thread(model.predict, live_obs, True)
                action_int = int(action) if hasattr(action, "__int__") else int(action.item())
                signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                signal     = signal_map.get(action_int, "HOLD")
                last_price = float(df["Close"].iloc[-1])
                last_rsi   = float(df["RSI"].iloc[-1])
                last_macd  = float(df["MACD"].iloc[-1])
                last_vol   = float(df["Volatility"].iloc[-1])

                # Position sizing from user cash balance
                live_state    = platform.get_state()
                cash_balance  = float(live_state.profile.cash_balance or 0)
                alloc_pct     = float(rl_alloc.value or 20) / 100.0
                budget_dollar = cash_balance * alloc_pct
                shares_calc   = max(0.0, round(budget_dollar / last_price, 2))

                # Save state for execute button
                rl_state.update({
                    "model": model, "ticker": ticker_sym, "win_size": win_size,
                    "df": df, "risk": risk_choice, "signal": signal,
                    "shares_to_trade": shares_calc, "price": last_price,
                })

                ret_color = lambda v: "#22c55e" if v >= 0 else "#ef4444"
                wr_color  = "#22c55e" if win_rate >= 50 else "#ef4444"

                rl_status.set_text(
                    f"✅ Trained on {len(train_df)} bars | Backtested on {len(test_df)} bars | "
                    f"Live signal on most recent bar"
                )

                rl_metrics.set_content(f"""
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0;">
                  <div style="background:#0d1117;border:1px solid #1e2530;border-radius:8px;padding:12px;border-top:2px solid {ret_color(agent_ret)}">
                    <div style="font-size:0.65rem;color:#64748b;font-family:monospace;letter-spacing:1px;">AGENT RETURN</div>
                    <div style="font-family:monospace;font-size:1.3rem;font-weight:600;color:{ret_color(agent_ret)}">{agent_ret:+.2f}%</div>
                  </div>
                  <div style="background:#0d1117;border:1px solid #1e2530;border-radius:8px;padding:12px;border-top:2px solid {ret_color(bh_ret)}">
                    <div style="font-size:0.65rem;color:#64748b;font-family:monospace;letter-spacing:1px;">BUY &amp; HOLD</div>
                    <div style="font-family:monospace;font-size:1.3rem;font-weight:600;color:{ret_color(bh_ret)}">{bh_ret:+.2f}%</div>
                  </div>
                  <div style="background:#0d1117;border:1px solid #1e2530;border-radius:8px;padding:12px;border-top:2px solid {ret_color(alpha)}">
                    <div style="font-size:0.65rem;color:#64748b;font-family:monospace;letter-spacing:1px;">ALPHA</div>
                    <div style="font-family:monospace;font-size:1.3rem;font-weight:600;color:{ret_color(alpha)}">{alpha:+.2f}%</div>
                  </div>
                  <div style="background:#0d1117;border:1px solid #1e2530;border-radius:8px;padding:12px;border-top:2px solid {wr_color}">
                    <div style="font-size:0.65rem;color:#64748b;font-family:monospace;letter-spacing:1px;">WIN RATE</div>
                    <div style="font-family:monospace;font-size:1.3rem;font-weight:600;color:{wr_color}">{win_rate:.1f}%</div>
                    <div style="font-size:0.7rem;color:#475569;">{wins}W / {losses}L</div>
                  </div>
                </div>
                """)

                # Live signal box
                signal_color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}[signal]
                signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal]
                rationale = ""
                if signal == "BUY":
                    rationale = (f"Agent detects upward momentum. RSI {last_rsi:.1f}, "
                                 f"MACD {last_macd:+.3f}, volatility {last_vol:.1f}%. "
                                 f"Entry recommended at ${last_price:.2f}.")
                elif signal == "SELL":
                    rationale = (f"Agent signals exit. RSI {last_rsi:.1f}, "
                                 f"MACD {last_macd:+.3f}. Either taking profits or cutting losses.")
                else:
                    rationale = (f"No clear edge. RSI {last_rsi:.1f}, MACD {last_macd:+.3f}. "
                                 f"Agent is staying out of the market on this bar.")

                rl_signal_box.set_content(f"""
                <div style="background:#0d1117;border:1px solid #1e2530;border-left:4px solid {signal_color};
                            border-radius:8px;padding:16px;margin-top:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                    <div>
                      <div style="font-size:0.65rem;color:#64748b;font-family:monospace;letter-spacing:2px;">LIVE SIGNAL</div>
                      <div style="font-family:monospace;font-size:1.6rem;font-weight:700;color:{signal_color};margin:4px 0;">
                        {signal_emoji} {signal} {ticker_sym} @ ${last_price:.2f}
                      </div>
                      <div style="font-size:0.85rem;color:#94a3b8;">{rationale}</div>
                    </div>
                    <div style="text-align:right;font-family:monospace;">
                      <div style="font-size:0.7rem;color:#64748b;letter-spacing:1px;">SUGGESTED TRADE</div>
                      <div style="font-size:1.1rem;color:#f1f5f9;margin-top:4px;">{shares_calc:.2f} shares</div>
                      <div style="font-size:0.85rem;color:#94a3b8;">≈ ${shares_calc*last_price:,.2f}</div>
                      <div style="font-size:0.7rem;color:#475569;margin-top:4px;">
                        ({alloc_pct*100:.0f}% of ${cash_balance:,.0f} cash)
                      </div>
                    </div>
                  </div>
                </div>
                """)

                # Show execute button only for actionable signals
                if signal in ("BUY", "SELL") and shares_calc > 0:
                    rl_execute_btn.classes(remove="hidden")
                    rl_execute_btn.set_text(f"📈 Execute: {signal} {shares_calc:.2f} {ticker_sym} @ ${last_price:.2f}")
                else:
                    rl_execute_btn.classes(add="hidden")

                if trades:
                    rows_html = "".join(
                        f'<tr><td style="color:{"#22c55e" if t["type"]=="WIN" else "#ef4444"};font-family:monospace;padding:4px 10px;">{t["type"]}</td>'
                        f'<td style="font-family:monospace;padding:4px 10px;">${t["entry"]}</td>'
                        f'<td style="font-family:monospace;padding:4px 10px;">${t["exit"]}</td>'
                        f'<td style="color:{"#22c55e" if t["pct"]>=0 else "#ef4444"};font-family:monospace;padding:4px 10px;">{t["pct"]:+.3f}%</td></tr>'
                        for t in trades[:20]
                    )
                    rl_chart.set_content(f"""
                    <div style="margin-top:12px;">
                      <div style="font-size:0.65rem;color:#475569;font-family:monospace;letter-spacing:2px;margin-bottom:8px;">BACKTEST TRADE LOG (last 20)</div>
                      <table style="width:100%;border-collapse:collapse;background:#0d1117;border:1px solid #1e2530;border-radius:6px;overflow:hidden;">
                        <thead><tr style="border-bottom:1px solid #1e2530;">
                          <th style="text-align:left;padding:6px 10px;font-size:0.65rem;color:#64748b;font-family:monospace;">OUTCOME</th>
                          <th style="text-align:left;padding:6px 10px;font-size:0.65rem;color:#64748b;font-family:monospace;">ENTRY</th>
                          <th style="text-align:left;padding:6px 10px;font-size:0.65rem;color:#64748b;font-family:monospace;">EXIT</th>
                          <th style="text-align:left;padding:6px 10px;font-size:0.65rem;color:#64748b;font-family:monospace;">RETURN</th>
                        </tr></thead>
                        <tbody>{rows_html}</tbody>
                      </table>
                    </div>
                    """)

            async def execute_rl_signal() -> None:
                if not rl_state["signal"] or rl_state["signal"] == "HOLD":
                    ui.notify("No actionable signal available — train first", type="warning")
                    return
                if rl_state["shares_to_trade"] <= 0:
                    ui.notify("Position size is zero — increase allocation %", type="warning")
                    return
                try:
                    await platform.execute_sim_trade(
                        rl_state["ticker"],
                        rl_state["signal"],
                        rl_state["shares_to_trade"],
                    )
                    refresh_all_sections()
                    ui.notify(
                        f"✅ Paper trade: {rl_state['signal']} {rl_state['shares_to_trade']:.2f} "
                        f"{rl_state['ticker']} @ ${rl_state['price']:.2f}",
                        type="positive",
                    )
                    rl_execute_btn.classes(add="hidden")
                except Exception as exc:
                    ui.notify(f"❌ Trade failed: {exc}", type="negative")

            rl_execute_btn.on_click(execute_rl_signal)
            ui.button("▶ Train & Generate Signal", on_click=run_rl_trainer).classes("self-start mt-2")

        # ── Connectors Panel ──────────────────────────────────────────────────
        with ui.card().classes("glass-card w-full"):
            ui.label("Connectors").classes("title-font text-xl")
            ui.label(
                "Sync live positions from Robinhood and bank transactions from Plaid. "
                "Configure credentials in your .env file to enable."
            ).classes("subtle mb-2")

            rh = platform.portfolio_service.robinhood
            pl = platform.portfolio_service.plaid

            with ui.row().classes("w-full gap-6 no-wrap max-[900px]:flex-wrap"):

                # Robinhood
                with ui.column().classes("gap-2 flex-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label("🟢 Robinhood" if rh.enabled else "⚪ Robinhood").classes("font-semibold")
                        ui.label("configured" if rh.enabled else "not configured").classes(
                            "neon-pill" if rh.enabled else "subtle"
                        )
                    ui.label("Syncs equity positions using ROBINHOOD_USERNAME + ROBINHOOD_PASSWORD.").classes("subtle")
                    rh_status = ui.label("").classes("subtle")

                    async def do_rh_sync(rh_status=rh_status) -> None:
                        if not rh.enabled:
                            rh_status.set_text("⚠ Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in .env first.")
                            return
                        rh_status.set_text("⏳ Syncing…")
                        result = await platform.sync_robinhood()
                        if result.ok:
                            rh_status.set_text(f"✅ Synced {result.positions_synced} position(s) at {result.synced_at[:19]}")
                            refresh_all_sections()
                        else:
                            rh_status.set_text(f"❌ {result.error or 'Sync failed'}")

                    ui.button("Sync Robinhood Now", on_click=do_rh_sync).props(
                        "flat" if not rh.enabled else "unelevated"
                    ).classes("self-start")

                # Plaid
                with ui.column().classes("gap-2 flex-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label("🟢 Plaid" if pl.enabled else "⚪ Plaid").classes("font-semibold")
                        ui.label("configured" if pl.enabled else "not configured").classes(
                            "neon-pill" if pl.enabled else "subtle"
                        )
                    ui.label("Pulls bank transactions using PLAID_CLIENT_ID + PLAID_SECRET + PLAID_ACCESS_TOKEN.").classes("subtle")
                    with ui.row().classes("items-center gap-2"):
                        ui.label("Days to sync:").classes("subtle")
                        plaid_days = ui.number(value=90, min=7, max=365, step=1).classes("w-20")
                    plaid_status = ui.label("").classes("subtle")
                    plaid_accounts_label = ui.label("").classes("subtle")

                    async def do_plaid_sync(plaid_status=plaid_status, plaid_accounts_label=plaid_accounts_label) -> None:
                        if not pl.enabled:
                            plaid_status.set_text("⚠ Set PLAID_CLIENT_ID, PLAID_SECRET, and PLAID_ACCESS_TOKEN in .env first.")
                            return
                        plaid_status.set_text("⏳ Syncing transactions…")
                        result = await platform.sync_plaid(days=int(plaid_days.value or 90))
                        if result.ok:
                            plaid_status.set_text(f"✅ {result.transactions_synced} transaction(s) synced at {result.synced_at[:19]}")
                            plaid_accounts_label.set_text(f"   {result.accounts_found} account(s) found")
                            refresh_all_sections()
                        else:
                            plaid_status.set_text(f"❌ {result.error or 'Sync failed'}")

                    async def do_plaid_accounts(plaid_accounts_label=plaid_accounts_label) -> None:
                        if not pl.enabled:
                            plaid_accounts_label.set_text("⚠ Plaid not configured.")
                            return
                        accounts = await pl.fetch_account_balances()
                        if accounts:
                            lines = [f"{a['name']} ({a['subtype']}): ${a['current_balance']:,.2f}" for a in accounts]
                            plaid_accounts_label.set_text(" | ".join(lines))
                        else:
                            plaid_accounts_label.set_text("No accounts returned.")

                    with ui.row().classes("gap-2"):
                        ui.button("Sync Transactions", on_click=do_plaid_sync).props(
                            "flat" if not pl.enabled else "unelevated"
                        ).classes("self-start")
                        ui.button("View Balances", on_click=do_plaid_accounts).props("flat").classes("self-start")

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
                    ui.button("Sim BUY", on_click=lambda: execute_trade("BUY"))
                    ui.button("Sim SELL", on_click=lambda: execute_trade("SELL"))
                ui.separator()
                ui.label("Scenario shock").classes("font-semibold")
                scenario_slider = ui.slider(min=-15, max=15, value=-5, step=1).classes("w-full")
                ui.button("Run scenario", on_click=update_scenario)

    def open_chat_popup() -> None:
        chat_container.set_visibility(True)
        chat_fab.set_visibility(False)

    def minimize_chat_popup() -> None:
        chat_container.set_visibility(False)
        chat_fab.set_visibility(True)

    chat_container = ui.card().classes("chat-card fixed bottom-4 right-4 z-50")
    with chat_container:
        with ui.row().classes("chat-header w-full"):
            with ui.column().classes("gap-1"):
                ui.label("Gemini AI Assistant").classes("title-font text-xl")
                ui.label("Ask anything about goals, tickers, risk, strategy, or your dashboard.").classes("subtle")
            with ui.row().classes("gap-1"):
                ui.button("−", on_click=minimize_chat_popup).props("flat round dense").tooltip("Minimize chat")
                ui.button("×", on_click=minimize_chat_popup).props("flat round dense").tooltip("Close chat")

        @ui.refreshable
        def chat_history() -> None:
            messages = platform.store.list_recent_chat_messages(platform.active_user_id, limit=20)
            if not messages:
                with ui.column().classes("chat-bubble border-cyan-400"):
                    ui.label("ASSISTANT").classes("metric-title text-xs")
                    ui.label("Hi — ask me any finance, ticker, risk, goal, or portfolio question. You are not limited to the suggested prompts.").classes("subtle")
                return
            for message in messages:
                css = "chat-bubble"
                if message.role == "assistant":
                    css += " border-cyan-400"
                with ui.column().classes(css):
                    ui.label(message.role.upper()).classes("metric-title text-xs")
                    ui.label(message.message).classes("subtle")
                    ui.label(message.timestamp.strftime("%H:%M:%S")).classes("metric-title text-xs")

        with ui.column().classes("chat-scroll w-full"):
            chat_history()
        typing_note = ui.label("AI is reasoning...").classes("typing subtle")
        typing_note.set_visibility(False)
        chat_input = ui.textarea("Ask any question — not just the suggestions").classes("chat-input-box w-full")
        chat_input.on("keydown.enter", send_chat)
        with ui.row().classes("w-full gap-2 items-center justify-between"):
            ui.label("Type any request. Suggestions are optional.").classes("subtle text-xs")
            ui.button("Send", on_click=send_chat)
        with ui.expansion("Optional suggested prompts", icon="tips_and_updates").classes("w-full"):
            for suggestion in platform.chat_service.suggested_prompts():
                ui.button(
                    suggestion,
                    on_click=lambda _, prompt=suggestion: send_chat(prompt),
                ).props("flat").classes("w-full")

    chat_container.set_visibility(False)
    chat_fab = ui.button("AI Chat", icon="smart_toy", on_click=open_chat_popup).classes("chat-fab fixed bottom-4 right-4 z-50")

    ui.timer(settings.refresh_seconds, lambda: asyncio.create_task(timed_refresh()))


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=settings.app_name,
        reload=False,
        favicon="💹",
        host="0.0.0.0",
        port=8080,
        storage_secret=settings.auth_secret_key,
    )
