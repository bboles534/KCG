from __future__ import annotations

from app.schemas import PortfolioSummary, ScenarioImpact, ScenarioResult


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

