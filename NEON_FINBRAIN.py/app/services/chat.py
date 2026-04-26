from __future__ import annotations

import re

from app.schemas import BrainDecision, ChatReply, NewsItem, PortfolioSummary, QuoteSnapshot, UserProfile
from app.store import SQLiteStore


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
