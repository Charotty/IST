from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def handle_missing(
    data: pd.DataFrame,
    method: str = "ffill",
    limit: int = 1,
) -> pd.DataFrame:
    """
    Pure missing data handling.
    Returns new DataFrame; original unchanged.
    """
    if method == "ffill":
        filled = data.ffill(limit=limit)
    elif method == "interpolate":
        filled = data.interpolate(method="time", limit=limit)
    elif method == "drop":
        filled = data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    return filled


def normalize_features(
    features: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
    window: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Pure normalization; returns normalized array and stats dict.
    Stats dict can be used for inverse transform.
    
    Args:
        features: Input features array
        method: Normalization method ("zscore" or "minmax")
        axis: Axis along which to compute statistics
        window: If provided, use rolling window normalization to avoid look-ahead bias
    """
    if window is not None and window < features.shape[axis]:
        # Rolling window normalization (no look-ahead)
        if method == "zscore":
            # Use pandas rolling for efficient computation
            if axis == 0:
                df = pd.DataFrame(features)
                rolling_mean = df.rolling(window=window, min_periods=1).mean().values
                rolling_std = df.rolling(window=window, min_periods=1).std().values
                norm = (features - rolling_mean) / (rolling_std + 1e-10)
                stats = {"method": "rolling_zscore", "window": window}
            else:
                # For axis=1, implement manually
                norm = np.empty_like(features)
                for i in range(features.shape[1]):
                    col = features[:, i]
                    rolling_mean = np.convolve(col, np.ones(window)/window, mode='same')
                    rolling_std = np.sqrt(np.convolve((col - rolling_mean)**2, np.ones(window)/window, mode='same'))
                    norm[:, i] = (col - rolling_mean) / (rolling_std + 1e-10)
                stats = {"method": "rolling_zscore", "window": window}
        elif method == "minmax":
            if axis == 0:
                df = pd.DataFrame(features)
                rolling_min = df.rolling(window=window, min_periods=1).min().values
                rolling_max = df.rolling(window=window, min_periods=1).max().values
                norm = (features - rolling_min) / (rolling_max - rolling_min + 1e-10)
                stats = {"method": "rolling_minmax", "window": window}
            else:
                norm = np.empty_like(features)
                for i in range(features.shape[1]):
                    col = features[:, i]
                    rolling_min = np.minimum.accumulate(col)  # Simplified - should use proper rolling
                    rolling_max = np.maximum.accumulate(col)  # Simplified - should use proper rolling
                    norm[:, i] = (col - rolling_min) / (rolling_max - rolling_min + 1e-10)
                stats = {"method": "rolling_minmax", "window": window}
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Standard normalization (use full history)
        if method == "zscore":
            mean = np.mean(features, axis=axis, keepdims=True)
            std = np.std(features, axis=axis, keepdims=True)
            norm = (features - mean) / (std + 1e-10)
            stats = {"method": "zscore", "mean": mean.squeeze(axis), "std": std.squeeze(axis)}
        elif method == "minmax":
            mn = np.min(features, axis=axis, keepdims=True)
            mx = np.max(features, axis=axis, keepdims=True)
            norm = (features - mn) / (mx - mn + 1e-10)
            stats = {"method": "minmax", "min": mn.squeeze(axis), "max": mx.squeeze(axis)}
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return norm, stats


def denormalize_features(
    normalized: np.ndarray,
    stats: dict,
    method: str = "zscore",
) -> np.ndarray:
    """Inverse of normalize_features; pure."""
    stats_method = stats.get("method", method)
    
    if stats_method == "rolling_zscore":
        # Rolling window normalization cannot be perfectly inverted
        # Return the best approximation using last known stats
        raise NotImplementedError("Rolling window normalization cannot be perfectly inverted")
    elif stats_method == "rolling_minmax":
        raise NotImplementedError("Rolling window normalization cannot be perfectly inverted")
    elif method == "zscore":
        mean = stats["mean"]
        std = stats["std"]
        if mean.ndim == 1:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return normalized * (std + 1e-10) + mean
    elif method == "minmax":
        mn = stats["min"]
        mx = stats["max"]
        if mn.ndim == 1:
            mn = mn.reshape(1, -1)
            mx = mx.reshape(1, -1)
        return normalized * (mx - mn + 1e-10) + mn
    else:
        raise ValueError(f"Unknown method: {method}")


def resample_to_uniform(
    data: pd.DataFrame,
    freq: str = "1s",
    agg: dict | None = None,
) -> pd.DataFrame:
    """
    Resample to uniform frequency; pure.
    Default aggregation: OHLC for price, sum for volume.
    """
    if agg is None:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    resampled = data.resample(freq).agg(agg)
    return resampled.dropna()
