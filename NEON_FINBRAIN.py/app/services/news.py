from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlparse

import feedparser
import yfinance as yf

from app.config import Settings
from app.schemas import NewsItem


@dataclass(slots=True)
class NewsCacheEntry:
    created_at: datetime
    items: list[NewsItem]


class NewsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, NewsCacheEntry] = {}

    async def get_news_for_ticker(self, ticker: str, force: bool = False) -> list[NewsItem]:
        cached = self._cache.get(ticker)
        if cached and not force and (datetime.now(UTC) - cached.created_at).total_seconds() < 300:
            return cached.items

        yahoo_items = await asyncio.to_thread(self._load_yahoo_news, ticker)
        rss_items = await asyncio.to_thread(self._load_google_rss, ticker)
        combined = self._dedupe(yahoo_items + rss_items)[: self.settings.max_news_items]
        self._cache[ticker] = NewsCacheEntry(created_at=datetime.now(UTC), items=combined)
        return combined

    async def get_news_for_tickers(
        self, tickers: list[str], force: bool = False
    ) -> dict[str, list[NewsItem]]:
        results = await asyncio.gather(
            *(self.get_news_for_ticker(ticker, force=force) for ticker in tickers),
            return_exceptions=True,
        )
        return {
            ticker: result
            for ticker, result in zip(tickers, results, strict=False)
            if not isinstance(result, Exception)
        }

    @staticmethod
    def _load_yahoo_news(ticker: str) -> list[NewsItem]:
        raw_items = getattr(yf.Ticker(ticker), "news", []) or []
        items: list[NewsItem] = []
        for item in raw_items:
            link = item.get("link") or ""
            source = item.get("publisher") or urlparse(link).netloc or "Yahoo Finance"
            timestamp = item.get("providerPublishTime")
            published_at = (
                datetime.fromtimestamp(timestamp, tz=UTC)
                if timestamp
                else datetime.now(UTC)
            )
            items.append(
                NewsItem(
                    ticker=ticker,
                    title=item.get("title") or f"{ticker} market update",
                    url=link,
                    source=source,
                    published_at=published_at,
                    summary=item.get("summary") or "",
                )
            )
        return items

    @staticmethod
    def _load_google_rss(ticker: str) -> list[NewsItem]:
        query = quote(f"{ticker} stock market")
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        )
        items: list[NewsItem] = []
        for entry in feed.entries[:8]:
            published_at = datetime.now(UTC)
            if entry.get("published"):
                try:
                    published_at = parsedate_to_datetime(entry.published).astimezone(UTC)
                except (TypeError, ValueError):
                    pass
            source = entry.get("source", {}).get("title") or urlparse(entry.link).netloc or "Google News"
            items.append(
                NewsItem(
                    ticker=ticker,
                    title=entry.title,
                    url=entry.link,
                    source=source,
                    published_at=published_at,
                    summary=entry.get("summary", ""),
                )
            )
        return items

    @staticmethod
    def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
        deduped: dict[str, NewsItem] = {}
        for item in sorted(items, key=lambda news: news.published_at, reverse=True):
            key = "".join(char for char in item.title.lower() if char.isalnum() or char.isspace()).strip()
            if key not in deduped:
                deduped[key] = item
        return list(deduped.values())

