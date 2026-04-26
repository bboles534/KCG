from __future__ import annotations

import asyncio
import re

import httpx

from app.config import Settings
from app.schemas import BrainDecision, ChatReply, NewsItem, PortfolioSummary, QuoteSnapshot, UserProfile
from app.store import SQLiteStore


class GeminiChatBot:
    """Gemini-powered chatbot for dashboard integration."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        self.conversation_history: list[dict] = []
    
    async def chat(self, message: str, context: dict | None = None) -> str:
        """Send a message to Gemini and get a response."""
        self.conversation_history.append({"role": "user", "parts": [{"text": message}]})
        
        system_context = self._build_system_context(context)
        full_prompt = f"{system_context}\n\nUser: {message}"
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                url = f"{self.base_url}:generateContent"
                params = {"key": self.api_key}
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": full_prompt}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 1024,
                    }
                }
                response = await client.post(url, params=params, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "candidates" in data and len(data["candidates"]) > 0:
                    ai_response = data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    ai_response = "I'm unable to process that request right now."
                
                self.conversation_history.append({"role": "model", "parts": [{"text": ai_response}]})
                return ai_response
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."
    
    def _build_system_context(self, context: dict | None) -> str:
        """Build system context from dashboard data."""
        if not context:
            return "You are a helpful financial assistant integrated into a trading dashboard."
        
        parts = ["You are an AI financial assistant with access to real-time market data."]
        
        if "portfolio" in context:
            portfolio = context["portfolio"]
            parts.append(f"Portfolio value: ${portfolio.get('total_value', 0):,.2f}")
            parts.append(f"Cash: ${portfolio.get('cash', 0):,.2f}")
            parts.append(f"Day PnL: ${portfolio.get('day_pnl', 0):,.2f}")
        
        if "market_regime" in context:
            parts.append(f"Current market regime: {context['market_regime']}")
        
        if "watchlist" in context:
            parts.append(f"User watchlist: {', '.join(context['watchlist'])}")
        
        if "recent_news" in context and context["recent_news"]:
            parts.append("Recent news highlights:")
            for news in context["recent_news"][:3]:
                parts.append(f"- {news}")
        
        return "\n".join(parts)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


class ChatService:
    TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

    def __init__(self, store: SQLiteStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings
        self.gemini_bot: GeminiChatBot | None = None
        
        if settings and settings.gemini_api_key:
            self.gemini_bot = GeminiChatBot(settings.gemini_api_key, settings.gemini_model)

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
        if profile_changed:
            self.store.save_user_profile(updated_profile)
            profile = updated_profile
        if any(word in message.lower() for word in ("goal", "today", "budget", "focus", "need")):
            self.store.add_daily_goal(user_id, message)

        tickers = self._extract_tickers(message, profile.watchlist)
        response = self._compose_response(message, profile, brain, portfolio, snapshots, news_map, tickers)
        self.store.add_chat_message(user_id, "assistant", response)
        return ChatReply(
            response=response,
            suggestions=self.suggested_prompts(),
            referenced_tickers=tickers,
            updated_profile=updated_profile if profile_changed else None,
        )
    
    async def gemini_chat(
        self,
        message: str,
        profile: UserProfile,
        brain: BrainDecision,
        portfolio: PortfolioSummary,
        snapshots: dict[str, QuoteSnapshot],
        news_map: dict[str, list[NewsItem]],
    ) -> str:
        """Handle chat via Gemini AI if available, fallback to rule-based."""
        if not self.gemini_bot:
            return self._compose_response(message, profile, brain, portfolio, snapshots, news_map, [])
        
        context = {
            "portfolio": {
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "day_pnl": portfolio.day_pnl,
            },
            "market_regime": brain.market_regime,
            "watchlist": profile.watchlist,
            "recent_news": [
                f"{ticker}: {item.title}" 
                for ticker, items in news_map.items() 
                for item in items[:1]
            ],
        }
        
        return await self.gemini_bot.chat(message, context)

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
                headline = news_map.get(ticker, [None])[0]
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
