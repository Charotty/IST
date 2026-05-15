"""Technical indicators for FeatureEngine (pandas-only)."""

from __future__ import annotations

import pandas as pd

from synchronization.indicators import adx, ema, rsi


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


__all__ = ["adx", "atr", "ema", "macd", "rsi"]
