from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import date

import httpx

from app.config import Settings
from app.schemas import ExpenseItem, PortfolioPosition, PortfolioSummary, QuoteSnapshot, UserProfile
from app.store import SQLiteStore


class RobinhoodConnector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.robinhood_username and self.settings.robinhood_password)

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
            username=self.settings.robinhood_username,
            password=self.settings.robinhood_password,
            store_session=False,
        )
        raw_positions = robinhood.account.build_holdings()
        positions: list[dict] = []
        for ticker, payload in raw_positions.items():
            positions.append(
                {
                    "ticker": ticker.upper(),
                    "shares": float(payload.get("quantity", 0)),
                    "avg_cost": float(payload.get("average_buy_price", 0)),
                    "thesis": "Synced from Robinhood",
                }
            )
        return positions


class PlaidConnector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.plaid_client_id
            and self.settings.plaid_secret
            and self.settings.plaid_access_token
        )

    async def fetch_expenses(self) -> list[dict]:
        if not self.enabled:
            return []
        end_date = date.today()
        start_date = date.fromordinal(end_date.toordinal() - 30)
        payload = {
            "client_id": self.settings.plaid_client_id,
            "secret": self.settings.plaid_secret,
            "access_token": self.settings.plaid_access_token,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{self.settings.plaid_base_url}/transactions/get",
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
    def __init__(self, store: SQLiteStore, settings: Settings) -> None:
        self.store = store
        self.robinhood = RobinhoodConnector(settings)
        self.plaid = PlaidConnector(settings)

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
        all_expenses = sorted(
            expenses + external_expenses,
            key=lambda item: item.incurred_on,
            reverse=True,
        )
        expenses_month = sum(item.amount for item in all_expenses)
        concentration = max((position.allocation for position in positions), default=0.0)
        diversification_score = max(0.0, 100 - concentration * 100 - max(len(positions) - 1, 0) * 2)
        risk_score = (
            sum(position.risk_score * position.allocation for position in positions) * 100
            if positions
            else 0.0
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
        spread_component = min(0.4, ((snapshot.spread or 0) / snapshot.last) * 100)
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
