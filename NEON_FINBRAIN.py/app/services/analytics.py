from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.schemas import Candle, TechnicalIndicators


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame([c.model_dump() for c in candles]).set_index("timestamp")
    return frame.sort_index()


def compute_volatility(candles: list[Candle]) -> float:
    frame = candles_to_frame(candles)
    if frame.empty or len(frame) < 5:
        return 0.0
    returns = frame["close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.tail(30).std() * math.sqrt(min(len(frame), 252)))


def compute_technicals(candles: list[Candle]) -> TechnicalIndicators:
    frame = candles_to_frame(candles)
    if frame.empty:
        return TechnicalIndicators()

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"].replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean().replace(0, np.nan)
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    sma_20 = close.rolling(20, min_periods=1).mean()
    sma_50 = close.rolling(50, min_periods=1).mean()

    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=1).mean()

    avg_volume = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
    volume_spike = (volume / avg_volume).fillna(1.0)

    latest_close = float(close.iloc[-1])
    latest_sma20 = float(sma_20.iloc[-1])
    latest_sma50 = float(sma_50.iloc[-1])
    latest_rsi = float(rsi.iloc[-1])
    latest_macd = float(macd.iloc[-1])
    latest_signal = float(macd_signal.iloc[-1])

    trend = "neutral"
    if latest_close > latest_sma20 > latest_sma50 and latest_macd >= latest_signal:
        trend = "bullish"
    elif latest_close < latest_sma20 < latest_sma50 and latest_macd <= latest_signal:
        trend = "bearish"

    patterns: list[str] = []
    if latest_rsi < 35 and latest_close >= latest_sma20 * 0.99:
        patterns.append("oversold-reversal")
    if latest_rsi > 68 and latest_close < latest_sma20:
        patterns.append("momentum-exhaustion")
    if len(close) > 20 and latest_close > float(close.tail(20).iloc[:-1].max()):
        patterns.append("breakout")
    if len(close) > 20 and latest_close < float(close.tail(20).iloc[:-1].min()):
        patterns.append("breakdown")
    if len(sma_20) > 2 and len(sma_50) > 2:
        previous_gap = float(sma_20.iloc[-2] - sma_50.iloc[-2])
        current_gap = float(sma_20.iloc[-1] - sma_50.iloc[-1])
        if previous_gap <= 0 < current_gap:
            patterns.append("golden-cross")
        if previous_gap >= 0 > current_gap:
            patterns.append("death-cross")
    if float(volume_spike.iloc[-1]) >= 1.8:
        patterns.append("volume-spike")

    return TechnicalIndicators(
        rsi=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_signal,
        sma_20=latest_sma20,
        sma_50=latest_sma50,
        ema_12=float(ema_12.iloc[-1]),
        ema_26=float(ema_26.iloc[-1]),
        atr=float(atr.iloc[-1]),
        volume_spike=float(volume_spike.iloc[-1]),
        trend=trend,
        patterns=patterns,
    )


def build_feature_frame(candles: list[Candle]) -> pd.DataFrame:
    frame = candles_to_frame(candles)
    if frame.empty or len(frame) < 40:
        return pd.DataFrame()

    close = frame["close"]
    volume = frame["volume"].replace(0, np.nan).fillna(method="ffill").fillna(0)
    technicals = compute_technicals(candles)

    feature_frame = pd.DataFrame(index=frame.index)
    feature_frame["return_1"] = close.pct_change()
    feature_frame["return_3"] = close.pct_change(3)
    feature_frame["return_10"] = close.pct_change(10)
    feature_frame["volatility_10"] = close.pct_change().rolling(10, min_periods=1).std()
    feature_frame["volume_ratio"] = volume / volume.rolling(20, min_periods=1).mean()
    feature_frame["sma_gap"] = (close / close.rolling(20, min_periods=1).mean()) - 1
    feature_frame["ema_gap"] = (close / close.ewm(span=12, adjust=False).mean()) - 1
    feature_frame["rsi_like"] = (
        close.diff().clip(lower=0).rolling(14, min_periods=1).mean()
        / close.diff().abs().rolling(14, min_periods=1).mean().replace(0, np.nan)
    ).fillna(0)
    feature_frame["atr_ratio"] = technicals.atr / close.replace(0, np.nan)
    feature_frame["future_return"] = close.shift(-3) / close - 1
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()
    return feature_frame

