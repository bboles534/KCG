from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


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


class ChatRequest(BaseModel):
    message: str


class PositionRequest(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    thesis: str = ""


class ExpenseRequest(BaseModel):
    category: str
    amount: float
    merchant: str = ""


class SimTradeRequest(BaseModel):
    ticker: str
    side: str
    shares: float


class ProfileUpdateRequest(BaseModel):
    investment_horizon: str
    risk_tolerance: str
    monthly_budget: float
    cash_balance: float
    watchlist: str

