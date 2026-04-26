from __future__ import annotations

from datetime import UTC, datetime

from app.schemas import BrainDecision, Insight, NewsItem, PortfolioSummary, QuoteSnapshot, UserProfile


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

        market_sentiment_score = sum(self._sentiment_score(items) for items in news_map.values()) / max(
            1, len(news_map)
        )
        regime = self._market_regime(scored, market_sentiment_score)
        risk_posture = self._risk_posture(profile, portfolio)
        confidence = min(
            0.95,
            0.5
            + sum(abs(score) for _, score, *_ in scored[:4]) / max(1, min(len(scored), 4)) * 0.22,
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

    def _opportunity_rationale(
        self,
        snapshot: QuoteSnapshot,
        sentiment_score: float,
        profile: UserProfile,
    ) -> str:
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

    def _warning_rationale(
        self,
        snapshot: QuoteSnapshot,
        sentiment_score: float,
        profile: UserProfile,
    ) -> str:
        return (
            f"Risk is rising from volatility ({snapshot.volatility:.2f}), spread "
            f"({((snapshot.spread or 0) / snapshot.last * 100) if snapshot.last else 0:.2f}%), "
            f"and sentiment ({sentiment_score:.2f}). That conflicts with a {profile.risk_tolerance} risk mandate."
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
        top_ticker, top_score, top_snapshot, *_ = scored[0]
        return (
            f"Market regime is {regime}. The strongest aligned setup is {top_ticker} with "
            f"{top_snapshot.prediction.probability_up:.0%} probability-up and {top_snapshot.technicals.trend} trend. "
            f"Portfolio risk is {portfolio.risk_score:.1f}/100, so recommendations are tuned for a "
            f"{profile.risk_tolerance} risk profile."
        )

    def _strategy_notes(self, profile: UserProfile, portfolio: PortfolioSummary, regime: str, scored) -> list[str]:
        notes = [
            f"Use {regime} positioning rules: press high-conviction names only when signal and sentiment agree.",
        ]
        if portfolio.expenses_month > profile.monthly_budget:
            notes.append("Spending is above budget, so favor smaller entries and preserve cash.")
        if profile.investment_horizon == "short-term":
            notes.append("For day trading, prioritize names with volume spike above 1.3x and tight spreads.")
        if profile.investment_horizon == "long-term":
            notes.append("For long-term accumulation, focus on bullish-trend names and average in across sessions.")
        if scored:
            notes.append(f"Highest priority watchlist name: {scored[0][0]}.")
        return notes

