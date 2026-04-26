"""
Feature Engineering Module

Enhanced features for trading models:
- Price lags (t-1, t-2, t-3, ...)
- Returns (log returns, percentage returns)
- Volatility (rolling std, ATR-like)
- Volume features
- Technical indicators (RSI, MACD, etc.)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List
import warnings


class FeatureEngineer:
    """Enhanced feature engineering for trading models."""

    def __init__(
        self,
        n_lags: int = 5,
        volatility_window: int = 10,
        include_volume: bool = True,
        include_indicators: bool = True,
        include_orderbook: bool = False
    ):
        self.n_lags = n_lags
        self.volatility_window = volatility_window
        self.include_volume = include_volume
        self.include_indicators = include_indicators
        self.include_orderbook = include_orderbook
        self.feature_names: List[str] = []
        self.orderbook_cache: List[Dict] = []  # Cache orderbook data

    def add_orderbook_features(self, orderbook: Dict[str, Any]) -> List[float]:
        """
        Extract features from orderbook data.

        Args:
            orderbook: Orderbook data from OKX with 'bids' and 'asks' arrays

        Returns:
            List of orderbook features
        """
        features = []
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if not bids or not asks:
            # Return zeros if no orderbook data
            return [0.0] * 10

        # Top level features
        best_bid_price = bids[0][0]
        best_ask_price = asks[0][0]
        best_bid_vol = bids[0][1]
        best_ask_vol = asks[0][1]

        # Bid-ask spread
        spread = best_ask_price - best_bid_price
        spread_pct = spread / best_bid_price if best_bid_price > 0 else 0

        # Mid price
        mid_price = (best_bid_price + best_ask_price) / 2

        # Order flow imbalance (OFI) - top level
        ofi = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol + 1e-10)

        # Top 5 levels bid/ask volumes
        top5_bid_vol = sum(b[1] for b in bids[:5])
        top5_ask_vol = sum(a[1] for a in asks[:5])

        # Volume imbalance across top 5 levels
        vol_imbalance_5 = (top5_bid_vol - top5_ask_vol) / (top5_bid_vol + top5_ask_vol + 1e-10)

        # Price pressure (distance from mid)
        bid_pressure = (mid_price - best_bid_price) / mid_price
        ask_pressure = (best_ask_price - mid_price) / mid_price

        # Order book depth (total volume)
        total_bid_vol = sum(b[1] for b in bids[:10])
        total_ask_vol = sum(a[1] for a in asks[:10])
        depth_ratio = total_bid_vol / (total_ask_vol + 1e-10)

        features.extend([
            spread_pct,  # Bid-ask spread percentage
            ofi,  # Order flow imbalance
            vol_imbalance_5,  # Volume imbalance top 5
            bid_pressure,  # Bid pressure
            ask_pressure,  # Ask pressure
            depth_ratio,  # Depth ratio
            best_bid_vol,  # Best bid volume
            best_ask_vol,  # Best ask volume
            top5_bid_vol,  # Top 5 bid volume
            top5_ask_vol,  # Top 5 ask volume
        ])

        return features

    def add_onchain_features(self, tx_volume: float, active_addresses: int, whale_score: float) -> List[float]:
        """
        Extract on-chain features.

        Args:
            tx_volume: Transaction volume in last period
            active_addresses: Number of active addresses
            whale_score: Whale movement score (0-1)

        Returns:
            List of on-chain features
        """
        features = [
            tx_volume,  # Transaction volume
            active_addresses,  # Active addresses
            whale_score,  # Whale movement score
            tx_volume / (active_addresses + 1e-10),  # Volume per address
        ]
        return features

    def build_features_from_ohlcv(
        self,
        ohlcv: np.ndarray,
        lookback: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build enhanced features from OHLCV data.

        Args:
            ohlcv: Array of shape (n, 5) with [open, high, low, close, volume]
            lookback: Number of previous candles to use for features

        Returns:
            X: Features array of shape (n - lookback, n_features)
            y: Target returns of shape (n - lookback,)
        """
        n = len(ohlcv)
        if n < lookback + 1:
            raise ValueError(f"Not enough data: need {lookback + 1}, got {n}")

        # Extract columns
        opens = ohlcv[:, 0]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]
        volumes = ohlcv[:, 4]

        # Calculate returns
        returns = np.zeros(n - 1)
        for i in range(n - 1):
            returns[i] = (closes[i + 1] - closes[i]) / closes[i]

        # Build features
        features_list = []
        self.feature_names = []

        for i in range(lookback, n - 1):
            window = ohlcv[i - lookback:i]
            row_features = []

            # Current price features
            current_close = closes[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_volume = volumes[i]

            # Basic price features
            row_features.extend([
                current_close,
                current_high - current_low,  # Range
                current_close - current_open,  # Body
                (current_close - current_open) / (current_high - current_low + 1e-10),  # Body ratio
            ])
            self.feature_names.extend([
                "close", "range", "body", "body_ratio"
            ])

            # Price lags
            for lag in range(1, self.n_lags + 1):
                if i - lag >= 0:
                    row_features.append(closes[i - lag])
                    self.feature_names.append(f"close_lag_{lag}")

            # Returns lags
            for lag in range(1, self.n_lags + 1):
                if i - lag >= 1:
                    row_features.append(returns[i - lag])
                    self.feature_names.append(f"return_lag_{lag}")

            # Volatility features (multi-period)
            for window_size in [10, 20, 30]:
                if i >= window_size:
                    window_returns = returns[i - window_size:i]
                    volatility = np.std(window_returns)
                    row_features.append(volatility)
                    self.feature_names.append(f"volatility_{window_size}")
                else:
                    row_features.append(0.0)
                    self.feature_names.append(f"volatility_{window_size}")

            # Mid price returns
            if i >= 1:
                mid_price = (current_high + current_low) / 2
                prev_mid = (highs[i-1] + lows[i-1]) / 2
                mid_return = (mid_price - prev_mid) / prev_mid
                row_features.append(mid_return)
                self.feature_names.append("mid_price_return")
            else:
                row_features.append(0.0)
                self.feature_names.append("mid_price_return")

            # Volume features
            if self.include_volume:
                row_features.extend([
                    current_volume,
                    current_volume / (np.mean(volumes[max(0, i - 10):i]) + 1e-10),  # Volume ratio
                ])
                self.feature_names.extend(["volume", "volume_ratio"])

                # Volume lags
                for lag in range(1, min(3, self.n_lags) + 1):
                    if i - lag >= 0:
                        row_features.append(volumes[i - lag])
                        self.feature_names.append(f"volume_lag_{lag}")

            # Technical indicators
            if self.include_indicators:
                # RSI (simplified)
                if i >= 14:
                    gains = np.where(np.diff(closes[i-14:i+1]) > 0, np.diff(closes[i-14:i+1]), 0)
                    losses = np.where(np.diff(closes[i-14:i+1]) < 0, -np.diff(closes[i-14:i+1]), 0)
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses) + 1e-10
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    row_features.append(rsi)
                    self.feature_names.append("rsi")
                else:
                    row_features.append(50.0)
                    self.feature_names.append("rsi")

                # MACD (simplified)
                if i >= 26:
                    ema_12 = self._calculate_ema(closes[i-26:i+1], 12)
                    ema_26 = self._calculate_ema(closes[i-26:i+1], 26)
                    macd = ema_12 - ema_26
                    row_features.append(macd)
                    self.feature_names.append("macd")
                else:
                    row_features.append(0.0)
                    self.feature_names.append("macd")

            features_list.append(row_features)

        X = np.array(features_list)
        y = returns[lookback:]  # Target: next return

        return X, y

    def build_features_with_extras(
        self,
        ohlcv: np.ndarray,
        orderbooks: Optional[List[Dict]] = None,
        onchain_data: Optional[List[Dict]] = None,
        lookback: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build features with orderbook and on-chain data.

        Args:
            ohlcv: Array of shape (n, 5) with [open, high, low, close, volume]
            orderbooks: List of orderbook data for each timestamp
            onchain_data: List of on-chain data for each timestamp
            lookback: Number of previous candles to use for features

        Returns:
            X: Features array of shape (n - lookback, n_features)
            y: Target returns of shape (n - lookback,)
        """
        n = len(ohlcv)
        if n < lookback + 1:
            raise ValueError(f"Not enough data: need {lookback + 1}, got {n}")

        # Extract columns
        opens = ohlcv[:, 0]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]
        volumes = ohlcv[:, 4]

        # Calculate returns
        returns = np.zeros(n - 1)
        for i in range(n - 1):
            returns[i] = (closes[i + 1] - closes[i]) / closes[i]

        # Build features
        features_list = []
        self.feature_names = []

        for i in range(lookback, n - 1):
            window = ohlcv[i - lookback:i]
            row_features = []

            # Current price features
            current_close = closes[i]
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_volume = volumes[i]

            # Basic price features
            row_features.extend([
                current_close,
                current_high - current_low,
                current_close - current_open,
                (current_close - current_open) / (current_high - current_low + 1e-10),
            ])
            self.feature_names.extend(["close", "range", "body", "body_ratio"])

            # Price lags
            for lag in range(1, self.n_lags + 1):
                if i - lag >= 0:
                    row_features.append(closes[i - lag])
                    self.feature_names.append(f"close_lag_{lag}")

            # Returns lags
            for lag in range(1, self.n_lags + 1):
                if i - lag >= 1:
                    row_features.append(returns[i - lag])
                    self.feature_names.append(f"return_lag_{lag}")

            # Volatility features (multi-period)
            for window_size in [10, 20, 30]:
                if i >= window_size:
                    window_returns = returns[i - window_size:i]
                    volatility = np.std(window_returns)
                    row_features.append(volatility)
                    self.feature_names.append(f"volatility_{window_size}")
                else:
                    row_features.append(0.0)
                    self.feature_names.append(f"volatility_{window_size}")

            # Mid price returns
            if i >= 1:
                mid_price = (current_high + current_low) / 2
                prev_mid = (highs[i-1] + lows[i-1]) / 2
                mid_return = (mid_price - prev_mid) / prev_mid
                row_features.append(mid_return)
                self.feature_names.append("mid_price_return")
            else:
                row_features.append(0.0)
                self.feature_names.append("mid_price_return")

            # Volume features
            if self.include_volume:
                row_features.extend([
                    current_volume,
                    current_volume / (np.mean(volumes[max(0, i - 10):i]) + 1e-10),
                ])
                self.feature_names.extend(["volume", "volume_ratio"])

            # Orderbook features
            if self.include_orderbook and orderbooks and i < len(orderbooks):
                ob_features = self.add_orderbook_features(orderbooks[i])
                row_features.extend(ob_features)
                self.feature_names.extend([
                    "spread_pct", "ofi", "vol_imbalance_5", "bid_pressure",
                    "ask_pressure", "depth_ratio", "best_bid_vol", "best_ask_vol",
                    "top5_bid_vol", "top5_ask_vol"
                ])
            elif self.include_orderbook:
                # Add zeros if no orderbook data
                row_features.extend([0.0] * 10)
                self.feature_names.extend([
                    "spread_pct", "ofi", "vol_imbalance_5", "bid_pressure",
                    "ask_pressure", "depth_ratio", "best_bid_vol", "best_ask_vol",
                    "top5_bid_vol", "top5_ask_vol"
                ])

            # On-chain features
            if onchain_data and i < len(onchain_data):
                oc = onchain_data[i]
                oc_features = self.add_onchain_features(
                    oc.get('tx_volume', 0),
                    oc.get('active_addresses', 0),
                    oc.get('whale_score', 0)
                )
                row_features.extend(oc_features)
                self.feature_names.extend([
                    "tx_volume", "active_addresses", "whale_score", "volume_per_address"
                ])
            else:
                # Add zeros if no on-chain data
                row_features.extend([0.0] * 4)
                self.feature_names.extend([
                    "tx_volume", "active_addresses", "whale_score", "volume_per_address"
                ])

            # Technical indicators
            if self.include_indicators:
                # RSI
                if i >= 14:
                    gains = np.where(np.diff(closes[i-14:i+1]) > 0, np.diff(closes[i-14:i+1]), 0)
                    losses = np.where(np.diff(closes[i-14:i+1]) < 0, -np.diff(closes[i-14:i+1]), 0)
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses) + 1e-10
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    row_features.append(rsi)
                    self.feature_names.append("rsi")
                else:
                    row_features.append(50.0)
                    self.feature_names.append("rsi")

                # MACD
                if i >= 26:
                    ema_12 = self._calculate_ema(closes[i-26:i+1], 12)
                    ema_26 = self._calculate_ema(closes[i-26:i+1], 26)
                    macd = ema_12 - ema_26
                    row_features.append(macd)
                    self.feature_names.append("macd")
                else:
                    row_features.append(0.0)
                    self.feature_names.append("macd")

            features_list.append(row_features)

        X = np.array(features_list)
        y = returns[lookback:]

        return X, y

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names


def prepare_classification_targets(
    returns: np.ndarray,
    threshold: float = 0.0015,
    use_three_classes: bool = True
) -> np.ndarray:
    """
    Convert returns to classification labels.

    Args:
        returns: Array of returns
        threshold: Threshold for BUY/SELL signals
        use_three_classes: If True, use 3 classes (SELL/HOLD/BUY), else 2 (SELL/BUY)

    Returns:
        Array of labels
    """
    if use_three_classes:
        labels = np.ones(len(returns), dtype=int)
        labels[returns < -threshold] = 0  # SELL
        labels[returns > threshold] = 2   # BUY
        # HOLD = 1
    else:
        labels = np.zeros(len(returns), dtype=int)
        labels[returns > 0] = 1  # BUY
        # SELL = 0

    return labels
