from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from its_project.features.base import BaseFeature


def _safe_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Pure RSI calculation without external dependencies."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.convolve(gains, np.ones(period), "valid") / period
    avg_loss = np.convolve(losses, np.ones(period), "valid") / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _safe_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Pure MACD calculation."""
    prices_series = pd.Series(prices)
    exp_fast = prices_series.ewm(span=fast, adjust=False).mean()
    exp_slow = prices_series.ewm(span=slow, adjust=False).mean()
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def _safe_bollinger(prices: np.ndarray, period: int = 20, std: float = 2.0) -> tuple:
    """Pure Bollinger Bands calculation."""
    sma = pd.Series(prices).rolling(window=period).mean()
    rolling_std = pd.Series(prices).rolling(window=period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    return upper.values, sma.values, lower.values


def _safe_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Pure ATR calculation."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr


def _safe_stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> tuple:
    """Pure Stochastic calculation."""
    lowest_low = pd.Series(low).rolling(window=k_period).min()
    highest_high = pd.Series(high).rolling(window=k_period).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=d_period).mean()
    return k_percent.values, d_percent.values


class TechnicalFeatures(BaseFeature):
    """Technical indicators (pure, immutable)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.indicators = config.get("indicators", ["rsi", "macd", "bbands", "atr", "stoch"])
        self.rsi_period = config.get("rsi_period", 14)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.atr_period = config.get("atr_period", 14)
        self.stoch_k = config.get("stoch_k", 14)
        self.stoch_d = config.get("stoch_d", 3)

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """Pure calculation; returns fixed-shape array."""
        self.validate_input(data)

        feats = {}
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        if "rsi" in self.indicators:
            rsi_vals = _safe_rsi(close, self.rsi_period)
            # Pad to original length
            rsi_full = np.full(len(close), np.nan)
            rsi_full[self.rsi_period :] = rsi_vals
            feats["rsi"] = rsi_full

        if "macd" in self.indicators:
            macd, signal, hist = _safe_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
            feats["macd"] = macd
            feats["macd_signal"] = signal
            feats["macd_hist"] = hist

        if "bbands" in self.indicators:
            upper, middle, lower = _safe_bollinger(close, self.bb_period, self.bb_std)
            feats["bb_upper"] = upper
            feats["bb_middle"] = middle
            feats["bb_lower"] = lower
            feats["bb_width"] = (upper - lower) / (middle + 1e-10)

        if "atr" in self.indicators:
            atr_vals = _safe_atr(high, low, close, self.atr_period)
            feats["atr"] = atr_vals

        if "stoch" in self.indicators:
            k_vals, d_vals = _safe_stoch(high, low, close, self.stoch_k, self.stoch_d)
            feats["stoch_k"] = k_vals
            feats["stoch_d"] = d_vals

        # Stack and fill NaNs
        self._feature_names = list(feats.keys())
        arr = np.column_stack([feats[name] for name in self._feature_names])
        arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()
