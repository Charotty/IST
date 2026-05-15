"""Align auxiliary series to the base timeframe grid (resample + fill)."""

from __future__ import annotations

import pandas as pd

from synchronization.config import SynchronizationConfig


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out = out.set_index("timestamp")
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def fill_series(series: pd.Series, method: str) -> pd.Series:
    if method == "ffill":
        return series.ffill()
    if method == "bfill":
        return series.bfill()
    if method == "linear":
        return series.interpolate(method="time")
    raise ValueError(f"Unsupported fill_method: {method!r}")


def align_to_base(
    df: pd.DataFrame,
    rule: str,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Resample auxiliary features to the base grid: ``resample(rule).last()`` then fill.

    Uses only completed auxiliary bars in each bin (no look-ahead within the bin).
    """
    aligned = ensure_datetime_index(df)
    resampled = aligned.resample(rule).last()
    return resampled.apply(lambda col: fill_series(col, fill_method))


def merge_auxiliary(
    base_df: pd.DataFrame,
    auxiliary: dict[str, pd.DataFrame],
    config: SynchronizationConfig,
) -> pd.DataFrame:
    """Join pre-aligned auxiliary feature frames onto ``base_df`` (left join on index)."""
    out = ensure_datetime_index(base_df)
    for _tf, features in auxiliary.items():
        aligned = align_to_base(features, config.rule, config.fill_method)
        out = out.join(aligned, how="left")
    if config.drop_na_after_merge:
        out = out.dropna()
    return out
