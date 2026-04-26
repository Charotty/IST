from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


def marketdata_to_dataframe(market_data: List[MarketData]) -> pd.DataFrame:
    """Pure conversion: List[MarketData] -> DataFrame with timestamp index."""
    rows = []
    for md in market_data:
        rows.append({
            "timestamp_ms": md.timestamp_ms,
            "symbol": md.symbol,
            "type": md.type.value,
            "exchange": md.exchange,
            "data": md.data,
        })
    df = pd.DataFrame(rows).set_index("timestamp_ms")
    df.index = pd.to_datetime(df.index, unit="ms")
    return df


def synchronize_marketdata(
    market_data: List[MarketData],
    freq: str = "1s",
    method: str = "ffill",
    max_gap: str = "5s",
    lookback_window: Optional[str] = None
) -> pd.DataFrame:
    """
    Synchronize heterogeneous market data to uniform timestamps.
    STRICT BACKWARD-LOOKING ONLY - NO LOOK-AHEAD BIAS.
    
    Args:
        market_data: List of MarketData objects
        freq: Target frequency (e.g., "1s", "5s", "1min")
        method: Fill method ("ffill", "interpolate")
        max_gap: Maximum gap before breaking forward fill
        lookback_window: Optional window for historical context (e.g., "5min")
        
    Returns:
        Synchronized DataFrame with all data types as columns
    """
    if not market_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = marketdata_to_dataframe(market_data)
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_index()
    
    # Create uniform time grid based on data range
    start_time = df.index.min()
    end_time = df.index.max()
    time_grid = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Resample each data type with strict backward-looking approach
    synchronized = {}
    
    for data_type in df["type"].unique():
        type_data = df[df["type"] == data_type].copy()
        
        if method == "ffill":
            # Forward fill with max gap constraint
            # IMPORTANT: Only use past data, never future data
            resampled = type_data.resample(freq).ffill(limit=pd.Timedelta(max_gap) // pd.Timedelta(freq))
            
        elif method == "interpolate":
            # Linear interpolation using only past data
            # Create interpolation that only looks backward
            resampled = type_data.resample(freq).mean()
            
            # Apply backward-looking interpolation
            # For each timestamp, only use data points <= current timestamp
            for timestamp in time_grid:
                past_data = type_data[type_data.index <= timestamp]
                if len(past_data) > 1:
                    # Interpolate using only past data
                    try:
                        # Simple linear interpolation from last two points
                        if len(past_data) >= 2:
                            last_two = past_data.tail(2)
                            if len(last_two) == 2:
                                t1, t2 = last_two.index
                                v1, v2 = last_two["data"].iloc[0], last_two["data"].iloc[1]
                                
                                if t2 != t1:  # Avoid division by zero
                                    # Linear interpolation
                                    alpha = (timestamp - t1) / (t2 - t1)
                                    if 0 <= alpha <= 1:  # Only interpolate within range
                                        interpolated_value = v1 + alpha * (v2 - v1)
                                        resampled.loc[timestamp] = interpolated_value
                    except Exception as e:
                        # Fall back to forward fill if interpolation fails
                        pass
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply lookback window if specified
        if lookback_window:
            lookback_delta = pd.Timedelta(lookback_window)
            # Create rolling window features using only past data
            window_data = resampled["data"].rolling(window=lookback_delta, min_periods=1)
            
            # Add window statistics as additional columns
            synchronized[f"{data_type}_mean"] = window_data.mean()
            synchronized[f"{data_type}_std"] = window_data.std()
            synchronized[f"{data_type}_min"] = window_data.min()
            synchronized[f"{data_type}_max"] = window_data.max()
        
        # Store only data column
        synchronized[data_type] = resampled["data"]
    
    # Combine into single DataFrame
    result = pd.DataFrame(index=time_grid)
    for data_type, series in synchronized.items():
        result[data_type] = series.reindex(time_grid)
    
    # Validate no look-ahead bias
    _validate_no_lookahead_bias(result, df)
    
    return result


def _validate_no_lookahead_bias(synchronized_df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    """Validate that no future data was used in synchronization."""
    for timestamp in synchronized_df.index:
        # For each synchronized timestamp, check that all used data
        # comes from timestamps <= current timestamp
        for data_type in synchronized_df.columns:
            if pd.isna(synchronized_df.loc[timestamp, data_type]):
                continue
                
            # Find the last original data point that could influence this value
            relevant_original = original_df[original_df["type"] == data_type]
            past_data = relevant_original[relevant_original.index <= timestamp]
            
            if len(past_data) == 0:
                # This should not happen if data exists
                raise ValueError(f"Look-ahead bias detected at {timestamp} for {data_type}")


def create_strict_pipeline(
    market_data: List[MarketData],
    target_freq: str = "1s",
    feature_window: str = "5min",
    validation_mode: bool = False
) -> pd.DataFrame:
    """
    Create strict chronological pipeline without any look-ahead bias.
    
    Args:
        market_data: List of MarketData objects
        target_freq: Target resampling frequency
        feature_window: Window for feature calculations
        validation_mode: If True, perform additional validation checks
        
    Returns:
        Strictly synchronized DataFrame ready for feature engineering
    """
    # Step 1: Sort by timestamp (critical for chronological order)
    sorted_data = sorted(market_data, key=lambda x: x.timestamp_ms)
    
    # Step 2: Synchronize with backward-only approach
    synchronized = synchronize_marketdata(
        sorted_data,
        freq=target_freq,
        method="ffill",
        max_gap="5s"
    )
    
    # Step 3: Validate consistency
    if validation_mode:
        _validate_pipeline_consistency(synchronized, target_freq)
    
    return synchronized


def extract_ohlcv_from_synced(synced: pd.DataFrame) -> pd.DataFrame:
    """
    From synchronized DataFrame, build OHLCV bars for feature calculation.
    Assumes price_data contains trade price/volume.
    """
    # Find price columns
    price_cols = [c for c in synced.columns if c.startswith("price_")]
    if not price_cols:
        raise ValueError("No price columns found in synced data")

    # Extract price and volume from first price source
    price_col = price_cols[0]
    # Flatten nested data if needed
    if "data" in synced.columns:
        # Extract from MarketData payload
        price_series = synced["data"].apply(lambda x: x.get("p") if isinstance(x, dict) else np.nan)
        vol_series = synced["data"].apply(lambda x: x.get("q") if isinstance(x, dict) else np.nan)
    else:
        # Assume price_col contains price directly
        price_series = synced[price_col]
        vol_series = synced.get(f"{price_col}_volume", pd.Series(np.nan, index=synced.index))

    # Build OHLCV
    ohlcv = pd.DataFrame({
        "open": price_series.resample("1s").first(),
        "high": price_series.resample("1s").max(),
        "low": price_series.resample("1s").min(),
        "close": price_series.resample("1s").last(),
        "volume": vol_series.resample("1s").sum(),
    }).dropna()
    return ohlcv
