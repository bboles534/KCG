from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from app.schemas import ChatTurn, ExpenseItem, SimulatedTrade, UserProfile


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

    def list_recent_goals(self, user_id: str, limit: int = 5) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT goal_text FROM daily_goals
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [row["goal_text"] for row in rows]

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
                SELECT role, message, created_at FROM chat_messages
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

    def upsert_position(
        self,
        user_id: str,
        ticker: str,
        shares: float,
        avg_cost: float,
        thesis: str = "",
    ) -> None:
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

    def add_expense(
        self,
        user_id: str,
        category: str,
        amount: float,
        merchant: str,
        incurred_on: date | None = None,
    ) -> None:
        expense_date = incurred_on or date.today()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO expense_entries
                    (user_id, category, amount, merchant, incurred_on)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    category,
                    amount,
                    merchant,
                    expense_date.isoformat(),
                ),
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

    def add_sim_trade(
        self,
        user_id: str,
        ticker: str,
        side: str,
        shares: float,
        price: float,
    ) -> None:
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

