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


def align_timestamps(
    data_sources: Dict[str, pd.DataFrame],
    freq: str = "1s",
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Pure function: align heterogeneous data to uniform time grid.
    Returns combined DataFrame with forward-filled or interpolated values.
    """
    # Union of all timestamps
    all_ts = set()
    for df in data_sources.values():
        all_ts.update(df.index)
    if not all_ts:
        return pd.DataFrame()

    # Uniform grid
    start, end = min(all_ts), max(all_ts)
    uniform_idx = pd.date_range(start=start, end=end, freq=freq)

    combined = pd.DataFrame(index=uniform_idx)

    for name, df in data_sources.items():
        if df.empty:
            continue
        # Resample to uniform grid
        if method == "ffill":
            resampled = df.resample(freq).ffill(limit=1)
        elif method == "interpolate":
            resampled = df.resample(freq).mean().interpolate(method="time", limit=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Prefix columns
        prefixed = resampled.add_prefix(f"{name}_")
        combined = combined.join(prefixed, how="outer")

    # Drop rows with all NaN
    combined = combined.dropna(how="all")
    return combined


def synchronize_marketdata(
    price_data: List[MarketData],
    lob_data: List[MarketData],
    onchain_data: List[MarketData],
    sentiment_data: List[MarketData],
    freq: str = "1s",
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Operator X_t = Phi(D^{LOB}_t, D^{price}_t, D^{onchain}_t, D^{sentiment}_t)
    Returns synchronized DataFrame on uniform 1s grid.
    """
    # Convert to DataFrames
    dfs = {}
    if price_data:
        dfs["price"] = marketdata_to_dataframe(price_data)
    if lob_data:
        dfs["lob"] = marketdata_to_dataframe(lob_data)
    if onchain_data:
        dfs["onchain"] = marketdata_to_dataframe(onchain_data)
    if sentiment_data:
        dfs["sentiment"] = marketdata_to_dataframe(sentiment_data)

    # Align to uniform grid
    synced = align_timestamps(dfs, freq=freq, method=method)
    return synced


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
