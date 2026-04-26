from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_csv(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    app_name: str = "Neon FinBrain"
    refresh_seconds: float = float(os.getenv("REFRESH_SECONDS", "3"))
    market_interval: str = os.getenv("MARKET_INTERVAL", "1m")
    market_period: str = os.getenv("MARKET_PERIOD", "5d")
    max_chart_points: int = int(os.getenv("MAX_CHART_POINTS", "240"))
    watchlist: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,TSLA,SPY,QQQ,AMD,AMZN")
        )
    )
    max_news_items: int = int(os.getenv("MAX_NEWS_ITEMS", "6"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")).resolve())
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", "data/financial_brain.db")).resolve()
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
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "demo-user")
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
