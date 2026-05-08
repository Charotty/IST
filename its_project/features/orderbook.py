from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from its_project.features.base import BaseFeature


def _ofi_from_levels(bids: np.ndarray, asks: np.ndarray, depth: int = 5) -> np.ndarray:
    """
    Pure Order Flow Imbalance calculation.
    bids/asks: shape (n_levels, 2) -> (price, volume)
    Returns OFI vector for each level.
    """
    if bids.shape[0] < depth or asks.shape[0] < depth:
        depth = min(bids.shape[0], asks.shape[0])
    bid_prices, bid_vols = bids[:depth, 0], bids[:depth, 1]
    ask_prices, ask_vols = asks[:depth, 0], asks[:depth, 1]

    # OFI = (delta_bid_vol * sign(delta_bid_price)) - (delta_ask_vol * sign(delta_ask_price))
    # For snapshot, use volume imbalance as proxy
    ofi = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-10)
    return ofi


def _spread(bids: np.ndarray, asks: np.ndarray) -> float:
    """Pure spread calculation (absolute and percentage)."""
    if bids.size == 0 or asks.size == 0:
        return np.nan
    best_bid = bids[0, 0]
    best_ask = asks[0, 0]
    return best_ask - best_bid


def _volume_imbalance(bids: np.ndarray, asks: np.ndarray, levels: int = 10) -> float:
    """Pure volume imbalance across top N levels."""
    n = min(levels, bids.shape[0], asks.shape[0])
    bid_vol = bids[:n, 1].sum()
    ask_vol = asks[:n, 1].sum()
    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / (total + 1e-10)


def _depth_imbalance(bids: np.ndarray, asks: np.ndarray, levels: int = 10) -> float:
    """Depth imbalance across available top-N levels."""
    if bids.size == 0 and asks.size == 0:
        return np.nan
    bid_levels = min(levels, bids.shape[0]) if bids.ndim == 2 else 0
    ask_levels = min(levels, asks.shape[0]) if asks.ndim == 2 else 0
    bid_volume = bids[:bid_levels, 1].sum() if bid_levels else 0.0
    ask_volume = asks[:ask_levels, 1].sum() if ask_levels else 0.0
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    return float((bid_volume - ask_volume) / total)


def _microprice(bids: np.ndarray, asks: np.ndarray) -> float:
    """Top-of-book microprice weighted by opposing queue volume."""
    if bids.size == 0 or asks.size == 0:
        return np.nan
    bid_price, bid_volume = bids[0, 0], bids[0, 1]
    ask_price, ask_volume = asks[0, 0], asks[0, 1]
    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return np.nan
    return float((bid_price * ask_volume + ask_price * bid_volume) / total_volume)


def _depth_measures(bids: np.ndarray, asks: np.ndarray, levels: int = 10) -> Dict[str, float]:
    """Pure depth measures (cumulative volume, weighted price)."""
    n = min(levels, bids.shape[0], asks.shape[0])
    bid_vol = bids[:n, 1].sum()
    ask_vol = asks[:n, 1].sum()
    bid_vwap = np.average(bids[:n, 0], weights=bids[:n, 1]) if bid_vol > 0 else np.nan
    ask_vwap = np.average(asks[:n, 0], weights=asks[:n, 1]) if ask_vol > 0 else np.nan
    return {
        "bid_depth": bid_vol,
        "ask_depth": ask_vol,
        "bid_vwap": bid_vwap,
        "ask_vwap": ask_vwap,
    }


def _price_density(bids: np.ndarray, asks: np.ndarray) -> Dict[str, float]:
    """Pure price density (levels per price interval)."""
    if bids.shape[0] < 2 or asks.shape[0] < 2:
        return {"bid_density": np.nan, "ask_density": np.nan}
    bid_range = bids[-1, 0] - bids[0, 0] + 1e-10
    ask_range = asks[-1, 0] - asks[0, 0] + 1e-10
    bid_density = bids.shape[0] / bid_range
    ask_density = asks.shape[0] / ask_range
    return {"bid_density": bid_density, "ask_density": ask_density}


class OrderBookFeatures(BaseFeature):
    """Order book-derived features (pure, immutable)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.depth_levels = config.get("depth_levels", 5)
        self.imbalance_levels = config.get("imbalance_levels", 10)

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Expects DataFrame with 'data' column containing orderbook dict.
        Returns fixed-shape array of features per timestamp.
        """
        required = ["data"]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Set feature names
        self._feature_names = self._get_feature_names()

        feats = {name: [] for name in self._feature_names}
        for _, row in data.iterrows():
            ob = row["data"]
            if not isinstance(ob, dict) or "bids" not in ob or "asks" not in ob:
                # Fill NaNs for missing orderbooks
                for name in feats:
                    feats[name].append(np.nan)
                continue

            bids = np.array(ob["bids"])
            asks = np.array(ob["asks"])

            # Spread
            spread = _spread(bids, asks)
            feats["spread"].append(spread)
            feats["spread_pct"].append(spread / (bids[0, 0] + 1e-10) * 100 if bids.size > 0 else np.nan)

            # Volume imbalance
            imb = _volume_imbalance(bids, asks, self.imbalance_levels)
            feats["volume_imbalance"].append(imb)

            # Depth measures
            depth = _depth_measures(bids, asks, self.imbalance_levels)
            feats["bid_depth"].append(depth["bid_depth"])
            feats["ask_depth"].append(depth["ask_depth"])
            feats["bid_vwap"].append(depth["bid_vwap"])
            feats["ask_vwap"].append(depth["ask_vwap"])

            # Price density
            dens = _price_density(bids, asks)
            feats["bid_density"].append(dens["bid_density"])
            feats["ask_density"].append(dens["ask_density"])

            # OFI vector (flattened)
            ofi = _ofi_from_levels(bids, asks, self.depth_levels)
            for i in range(self.depth_levels):
                feats[f"ofi_{i}"].append(ofi[i] if i < len(ofi) else np.nan)

        # Convert to arrays and handle NaNs
        arr = np.column_stack([np.array(feats[name], dtype=float) for name in self._feature_names])
        arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def _get_feature_names(self) -> List[str]:
        """Helper to generate consistent feature names."""
        names = [
            "spread",
            "spread_pct",
            "volume_imbalance",
            "bid_depth",
            "ask_depth",
            "bid_vwap",
            "ask_vwap",
            "bid_density",
            "ask_density",
        ]
        for i in range(self.depth_levels):
            names.append(f"ofi_{i}")
        return names
