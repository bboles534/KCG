from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import yfinance as yf

from app.config import Settings
from app.schemas import Candle, QuoteSnapshot
from app.services.analytics import compute_technicals, compute_volatility
from app.services.ml_engine import PredictionEngine


@dataclass(slots=True)
class CacheEntry:
    snapshot: QuoteSnapshot
    created_at: datetime


class YahooFinanceProvider:
    name = "yfinance"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_snapshot(self, ticker: str) -> QuoteSnapshot:
        return await asyncio.to_thread(self._get_snapshot_sync, ticker)

    def _get_snapshot_sync(self, ticker: str) -> QuoteSnapshot:
        instrument = yf.Ticker(ticker)
        history = instrument.history(
            period=self.settings.market_period,
            interval=self.settings.market_interval,
            auto_adjust=False,
            prepost=True,
        )
        if history.empty:
            raise ValueError(f"No data returned for {ticker}")

        history = history.tail(self.settings.max_chart_points)
        candles = self._to_candles(history)
        fast_info = dict(getattr(instrument, "fast_info", {}) or {})
        last = float(fast_info.get("lastPrice") or history["Close"].iloc[-1])
        previous_close = float(
            fast_info.get("previousClose")
            or history["Close"].iloc[-2]
            if len(history) > 1
            else last
        )
        bid = _to_optional_float(fast_info.get("bid"))
        ask = _to_optional_float(fast_info.get("ask"))
        spread = ask - bid if bid is not None and ask is not None else None
        volume = float(fast_info.get("lastVolume") or history["Volume"].iloc[-1] or 0)
        avg_volume = float(history["Volume"].tail(30).mean() or 0)
        change = last - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0.0

        return QuoteSnapshot(
            ticker=ticker,
            source=self.name,
            last=last,
            previous_close=previous_close,
            change=change,
            change_percent=change_percent,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=volume,
            avg_volume=avg_volume,
            volatility=compute_volatility(candles),
            last_updated=datetime.now(UTC),
            candles=candles,
            technicals=compute_technicals(candles),
        )

    @staticmethod
    def _to_candles(frame: pd.DataFrame) -> list[Candle]:
        candles: list[Candle] = []
        for timestamp, row in frame.iterrows():
            ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            candles.append(
                Candle(
                    timestamp=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"] or 0),
                )
            )
        return candles


class PolygonProvider:
    name = "polygon"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def get_snapshot(self, ticker: str) -> QuoteSnapshot:
        end = datetime.now(UTC)
        start = end - timedelta(days=5)
        bars_url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
            f"{start.date()}/{end.date()}?adjusted=true&sort=asc&limit=500&apiKey={self.api_key}"
        )
        trade_url = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={self.api_key}"
        quote_url = f"https://api.polygon.io/v2/last/nbbo/{ticker}?apiKey={self.api_key}"
        async with httpx.AsyncClient(timeout=10) as client:
            bars_resp, trade_resp, quote_resp = await asyncio.gather(
                client.get(bars_url),
                client.get(trade_url),
                client.get(quote_url),
            )
        bars_resp.raise_for_status()
        trade_resp.raise_for_status()
        quote_resp.raise_for_status()
        bars_data = bars_resp.json().get("results", [])
        trade_data = trade_resp.json().get("results", {})
        quote_data = quote_resp.json().get("results", {})
        if not bars_data:
            raise ValueError(f"No polygon bars for {ticker}")

        candles = [
            Candle(
                timestamp=datetime.fromtimestamp(item["t"] / 1000, tz=UTC),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
            )
            for item in bars_data[-240:]
        ]
        last = float(trade_data.get("p") or candles[-1].close)
        previous_close = candles[-2].close if len(candles) > 1 else last
        bid = _to_optional_float(quote_data.get("P"))
        ask = _to_optional_float(quote_data.get("p"))
        spread = ask - bid if bid is not None and ask is not None else None
        return QuoteSnapshot(
            ticker=ticker,
            source=self.name,
            last=last,
            previous_close=previous_close,
            change=last - previous_close,
            change_percent=((last - previous_close) / previous_close * 100) if previous_close else 0.0,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=candles[-1].volume,
            avg_volume=sum(c.volume for c in candles[-30:]) / min(30, len(candles)),
            volatility=compute_volatility(candles),
            last_updated=datetime.now(UTC),
            candles=candles,
            technicals=compute_technicals(candles),
        )


class AlpacaProvider:
    name = "alpaca"

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

    async def get_snapshot(self, ticker: str) -> QuoteSnapshot:
        end = datetime.now(UTC)
        start = end - timedelta(days=2)
        bars_url = (
            f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
            f"?timeframe=1Min&start={start.isoformat()}&end={end.isoformat()}&limit=240"
        )
        quotes_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest"
        async with httpx.AsyncClient(timeout=10, headers=self.headers) as client:
            bars_resp, quote_resp = await asyncio.gather(client.get(bars_url), client.get(quotes_url))
        bars_resp.raise_for_status()
        quote_resp.raise_for_status()
        bars = bars_resp.json().get("bars", [])
        latest_quote = quote_resp.json().get("quote", {})
        if not bars:
            raise ValueError(f"No alpaca bars for {ticker}")
        candles = [
            Candle(
                timestamp=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
            )
            for item in bars
        ]
        last = candles[-1].close
        previous_close = candles[-2].close if len(candles) > 1 else last
        bid = _to_optional_float(latest_quote.get("bp"))
        ask = _to_optional_float(latest_quote.get("ap"))
        spread = ask - bid if bid is not None and ask is not None else None
        return QuoteSnapshot(
            ticker=ticker,
            source=self.name,
            last=last,
            previous_close=previous_close,
            change=last - previous_close,
            change_percent=((last - previous_close) / previous_close * 100) if previous_close else 0.0,
            bid=bid,
            ask=ask,
            spread=spread,
            volume=candles[-1].volume,
            avg_volume=sum(c.volume for c in candles[-30:]) / min(30, len(candles)),
            volatility=compute_volatility(candles),
            last_updated=datetime.now(UTC),
            candles=candles,
            technicals=compute_technicals(candles),
        )


class MarketDataService:
    def __init__(self, settings: Settings, predictor: PredictionEngine) -> None:
        self.settings = settings
        self.predictor = predictor
        self._cache: dict[str, CacheEntry] = {}
        self.providers = []
        if settings.polygon_api_key:
            self.providers.append(PolygonProvider(settings.polygon_api_key))
        if settings.alpaca_api_key and settings.alpaca_api_secret:
            self.providers.append(AlpacaProvider(settings.alpaca_api_key, settings.alpaca_api_secret))
        self.providers.append(YahooFinanceProvider(settings))

    async def get_snapshot(self, ticker: str, force: bool = False) -> QuoteSnapshot:
        cached = self._cache.get(ticker)
        if (
            cached
            and not force
            and (datetime.now(UTC) - cached.created_at).total_seconds() < self.settings.refresh_seconds
        ):
            return cached.snapshot

        last_error: Exception | None = None
        for provider in self.providers:
            try:
                snapshot = await provider.get_snapshot(ticker)
                snapshot.prediction = self.predictor.predict(ticker, snapshot.candles)
                self._cache[ticker] = CacheEntry(snapshot=snapshot, created_at=datetime.now(UTC))
                return snapshot
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Unable to load market data for {ticker}: {last_error}")

    async def get_snapshots(self, tickers: list[str], force: bool = False) -> dict[str, QuoteSnapshot]:
        tasks = [self.get_snapshot(ticker, force=force) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: dict[str, QuoteSnapshot] = {}
        for ticker, result in zip(tickers, results, strict=False):
            if isinstance(result, QuoteSnapshot):
                snapshots[ticker] = result
        return snapshots


def _to_optional_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

