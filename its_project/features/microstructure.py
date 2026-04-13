from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from its_project.features.base import BaseFeature


def _roll_effective_spread(prices: np.ndarray) -> float:
    """
    Pure Roll (1984) effective spread estimator.
    Returns cost of round-trip transaction.
    """
    if len(prices) < 2:
        return np.nan
    price_changes = np.diff(prices)
    # Covariance of price changes with lag 1
    cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
    roll = -2 * cov
    return roll


def _vpin(buy_volume: np.ndarray, sell_volume: np.ndarray) -> float:
    """
    Pure VPIN (Volume-Synchronized Probability of Informed Trading).
    Normalized absolute order flow imbalance.
    """
    total_volume = buy_volume + sell_volume
    if total_volume == 0:
        return np.nan
    vpin = np.abs(buy_volume - sell_volume) / total_volume
    return vpin


def _realized_volatility(returns: np.ndarray) -> float:
    """Pure realized volatility (sqrt of sum of squared returns)."""
    if len(returns) == 0:
        return np.nan
    return np.sqrt(np.sum(returns ** 2))


def _amihud_illiquidity(returns: np.ndarray, volumes: np.ndarray) -> float:
    """
    Pure Amihud (2002) illiquidity measure.
    Average of |return| / volume.
    """
    if len(returns) == 0 or len(volumes) == 0:
        return np.nan
    # Avoid division by zero
    mask = volumes > 0
    if not np.any(mask):
        return np.nan
    illiq = np.mean(np.abs(returns[mask]) / volumes[mask])
    return illiq


def _kyle_lambda(order_flow: np.ndarray, returns: np.ndarray) -> float:
    """
    Pure Kyle (1985) lambda: price impact coefficient.
    Regression of returns on order flow.
    """
    if len(order_flow) < 2 or len(returns) < 2:
        return np.nan
    # Simple linear regression slope
    X = order_flow.reshape(-1, 1)
    y = returns
    # Avoid singular matrix
    if np.var(X) == 0:
        return np.nan
    lambda_coef = np.cov(X.flatten(), y)[0, 1] / np.var(X)
    return lambda_coef


class MicrostructureFeatures(BaseFeature):
    """Market microstructure features (pure, immutable)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.window = config.get("window", 20)  # Rolling window for estimators

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Expects DataFrame with OHLCV.
        Returns fixed-shape array of microstructure features.
        """
        self.validate_input(data)

        # Compute returns and volumes
        prices = data["close"].values
        volumes = data["volume"].values
        returns = np.log(prices / np.roll(prices, 1))[1:]  # Drop first NaN

        # Rolling window calculations
        n = len(prices)
        feats = {name: np.full(n, np.nan) for name in self._feature_names()}

        for i in range(self.window, n):
            window_prices = prices[i - self.window : i + 1]
            window_returns = returns[i - self.window : i]
            window_volumes = volumes[i - self.window : i]

            # Roll effective spread
            roll_val = _roll_effective_spread(window_prices)
            feats["roll"][i] = roll_val

            # Realized volatility
            rv = _realized_volatility(window_returns)
            feats["realized_vol"][i] = rv

            # Amihud illiquidity
            amihud = _amihud_illiquidity(window_returns, window_volumes)
            feats["amihud"][i] = amihud

            # VPIN (proxy using volume imbalance)
            # Assume buy volume when price goes up, sell when down
            price_changes = np.diff(window_prices)
            buy_vol = np.where(price_changes > 0, window_volumes[1:], 0)
            sell_vol = np.where(price_changes < 0, window_volumes[1:], 0)
            vpin_val = _vpin(buy_vol, sell_vol)
            feats["vpin"][i] = vpin_val

            # Kyle lambda (order flow proxy = volume * price change)
            order_flow = window_volumes[1:] * price_changes
            if len(order_flow) > 1 and len(window_returns) > 1:
                kyle_val = _kyle_lambda(order_flow, window_returns[1:])
                feats["kyle_lambda"][i] = kyle_val

        # Fill NaNs at beginning
        for name in feats:
            feats[name][: self.window] = 0.0

        arr = np.column_stack([feats[name] for name in self._feature_names()])
        return arr

    def get_feature_names(self) -> List[str]:
        return self._feature_names()

    def _feature_names(self) -> List[str]:
        """Helper to generate consistent feature names."""
        return [
            "roll",
            "realized_vol",
            "amihud",
            "vpin",
            "kyle_lambda",
        ]
