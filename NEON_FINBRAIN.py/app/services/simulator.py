from __future__ import annotations

from collections import defaultdict

from app.schemas import QuoteSnapshot, SimulationBook, SimulationPosition
from app.store import SQLiteStore


class PaperTradingService:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def execute_trade(
        self,
        user_id: str,
        ticker: str,
        side: str,
        shares: float,
        price: float,
    ) -> None:
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

