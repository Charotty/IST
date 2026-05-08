from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BasicFeatures:
    """Basic price features: returns, log returns, lags, and volatility."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.features = config.get("features", ["returns", "log_returns", "lags"])
        self.return_periods = config.get("return_periods", [1, 5, 15])
        self._return_periods_configured = "return_periods" in config
        self.lag_periods = config.get("lag_periods", [1, 2, 3, 5, 10])
        self._lag_periods_configured = "lag_periods" in config
        self.volatility_windows = config.get("volatility_windows", [10, 20, 30])
        self._volatility_windows_configured = "volatility_windows" in config
        self.atr_periods = config.get("atr_periods", [14, 21])
        self.price_column = config.get("price_column", "close")
        self._price_column_configured = "price_column" in config
        self._feature_names: list[str] = []

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if (
            not self._price_column_configured
            and self.price_column not in data.columns
            and "price" in data.columns
        ):
            price_column = "price"
        else:
            price_column = self.price_column

        if price_column not in data.columns:
            raise KeyError(price_column)

        if not self.features:
            self._feature_names = []
            return pd.DataFrame(index=data.index)

        price = data[price_column]
        outputs: list[pd.Series] = []
        names: list[str] = []

        if "returns" in self.features:
            for period in self._active_return_periods():
                denominator = price.shift(period)
                returns = (price - denominator) / denominator
                returns = returns.mask((denominator == 0) | (price == 0))
                outputs.append(returns.rename(f"return_{period}"))
                names.append(f"return_{period}")

        if "log_returns" in self.features:
            for period in self._active_return_periods():
                previous = price.shift(period)
                valid = (price > 0) & (previous > 0)
                log_returns = pd.Series(np.nan, index=data.index, dtype=float)
                log_returns[valid] = np.log(price[valid] / previous[valid])
                outputs.append(log_returns.rename(f"log_return_{period}"))
                names.append(f"log_return_{period}")

        if "lags" in self.features:
            for period in self._active_lag_periods():
                outputs.append(price.shift(period).rename(f"lag_{period}"))
                names.append(f"lag_{period}")

        if "volatility" in self.features:
            if self._volatility_windows_configured or not {"high", "low", "close"}.issubset(data.columns):
                for window in self.volatility_windows:
                    outputs.append(price.shift(1).rolling(window=window).std().rename(f"rolling_std_{window}"))
                    names.append(f"rolling_std_{window}")

            has_ohlc = {"high", "low", "close"}.issubset(data.columns)
            if has_ohlc:
                for period in self.atr_periods:
                    outputs.append(self._atr(data, period).rename(f"atr_{period}"))
                    names.append(f"atr_{period}")

        self._feature_names = names
        if not outputs:
            return pd.DataFrame(index=data.index)
        return pd.concat(outputs, axis=1)

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()

    def _active_return_periods(self) -> list[int]:
        return self.return_periods if self._return_periods_configured else [1]

    def _active_lag_periods(self) -> list[int]:
        return self.lag_periods if self._lag_periods_configured else [1]

    @staticmethod
    def _atr(data: pd.DataFrame, period: int) -> pd.Series:
        high = data["high"]
        low = data["low"]
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        return true_range.rolling(window=period).mean()
