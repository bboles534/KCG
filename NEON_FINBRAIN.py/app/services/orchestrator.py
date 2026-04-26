from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.config import Settings
from app.schemas import BrainDecision, PortfolioSummary, QuoteSnapshot, ScenarioResult, SimulationBook, UserProfile
from app.services.brain import BrainService
from app.services.chat import ChatService
from app.services.market_data import MarketDataService
from app.services.ml_engine import PredictionEngine
from app.services.news import NewsService
from app.services.portfolio import PortfolioService
from app.services.scenarios import ScenarioEngine
from app.services.sentiment import SentimentEngine
from app.services.simulator import PaperTradingService
from app.store import SQLiteStore


@dataclass(slots=True)
class DashboardState:
    profile: UserProfile
    market: dict[str, QuoteSnapshot] = field(default_factory=dict)
    news: dict[str, list] = field(default_factory=dict)
    portfolio: PortfolioSummary = field(default_factory=PortfolioSummary)
    brain: BrainDecision = field(default_factory=BrainDecision)
    simulation: SimulationBook = field(default_factory=SimulationBook)
    scenario: ScenarioResult | None = None
    refreshed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PlatformOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = SQLiteStore(settings.db_path)
        self.predictor = PredictionEngine()
        self.market_service = MarketDataService(settings, self.predictor)
        self.news_service = NewsService(settings)
        self.sentiment_engine = SentimentEngine(settings)
        self.portfolio_service = PortfolioService(self.store, settings)
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
        profile = self.store.load_user_profile(self.settings.default_user_id, self.settings.watchlist)
        self.state = DashboardState(profile=profile)
        self._bootstrapped = True
        await self.refresh_all(force=True)

    async def ensure_fresh(self) -> None:
        await self.ensure_bootstrapped()
        assert self.state is not None
        age = (datetime.now(UTC) - self.state.refreshed_at).total_seconds()
        if age >= self.settings.refresh_seconds:
            await self.refresh_all()

    async def refresh_all(self, force: bool = False) -> DashboardState:
        await self.ensure_bootstrapped()
        async with self._lock:
            assert self.state is not None
            profile = self.store.load_user_profile(self.settings.default_user_id, self.settings.watchlist)
            market, news = await asyncio.gather(
                self.market_service.get_snapshots(profile.watchlist, force=force),
                self.news_service.get_news_for_tickers(profile.watchlist[:6], force=force),
            )
            enriched_news = {
                ticker: self.sentiment_engine.enrich(items)
                for ticker, items in news.items()
            }
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
        assert self.state is not None
        profile = self.store.load_user_profile(self.settings.default_user_id, self.settings.watchlist)
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
        self.store.upsert_position(self.settings.default_user_id, ticker, shares, avg_cost, thesis)
        return await self.refresh_all(force=True)

    async def add_expense(self, category: str, amount: float, merchant: str) -> DashboardState:
        await self.ensure_bootstrapped()
        self.store.add_expense(self.settings.default_user_id, category, amount, merchant)
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
        self.simulator.execute_trade(
            self.settings.default_user_id,
            ticker,
            side,
            shares,
            snapshot.last,
        )
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

    async def chat(self, message: str):
        await self.ensure_fresh()
        state = self.get_state()
        reply = self.chat_service.handle_message(
            self.settings.default_user_id,
            message,
            state.profile,
            state.brain,
            state.portfolio,
            state.market,
            state.news,
        )
        refreshed = await self.refresh_all()
        return reply, refreshed
