"""Base OHLCV feature computation (``FeatureEngine`` from ``ist.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.config import BaseIndicatorsConfig, FeatureEngineeringConfig
from feature_engineering.indicators import adx, atr, ema, macd, rsi
from synchronization.gap_handler import ensure_datetime_index


class FeatureEngine:
    """Add trend, momentum, volatility, and regime indicators to OHLCV (+ preserved MTF cols)."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: FeatureEngineeringConfig | None = None,
    ):
        self.config = config or FeatureEngineeringConfig()
        self.params = self.config.base_indicators
        self.df = ensure_datetime_index(df)

    def add_indicators(self) -> "FeatureEngine":
        """Add base indicators; MTF columns already on ``df`` are kept."""
        p = self.params
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]

        self.df["ema_fast"] = ema(close, p.ema_fast)
        self.df["ema_slow"] = ema(close, p.ema_slow)
        self.df["ema_slope"] = (self.df["ema_fast"] - self.df["ema_fast"].shift(1)) / self.df[
            "ema_fast"
        ].shift(1)

        self.df["rsi"] = rsi(close, p.rsi_length)

        macd_line, macd_signal, macd_hist = macd(
            close, p.macd_fast, p.macd_slow, p.macd_signal
        )
        self.df["macd"] = macd_line
        self.df["macd_signal"] = macd_signal
        self.df["macd_hist"] = macd_hist

        self.df["atr"] = atr(high, low, close, p.atr_length)
        self.df["log_ret"] = np.log(close / close.shift(1))
        self.df["volatility"] = (
            self.df["log_ret"].rolling(window=p.vol_window).std() * p.vol_annualize_factor
        )

        self.df["adx"] = adx(high, low, close, p.adx_length)
        return self

    def get_processed_data(self) -> pd.DataFrame:
        if self.config.drop_na:
            return self.df.dropna()
        return self.df
