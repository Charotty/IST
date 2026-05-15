"""OHLCV validation: duplicates, end-date trim, gap detection."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to UTC DatetimeIndex named ``timestamp``."""
    out = df.copy()
    if "timestamp" in out.columns:
        ts = out["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            out["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            out["timestamp"] = pd.to_datetime(ts, utc=True)
        out = out.set_index("timestamp")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    out.index.name = "timestamp"
    return out.sort_index()


def drop_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate bars; keep the last row per timestamp."""
    out = ensure_datetime_index(df)
    return out[~out.index.duplicated(keep="last")]


def trim_to_end_date(df: pd.DataFrame, end_str: str) -> pd.DataFrame:
    """Keep rows with timestamp <= end_date (inclusive)."""
    out = ensure_datetime_index(df)
    end = pd.to_datetime(end_str, utc=True)
    return out[out.index <= end]


def detect_gaps(df: pd.DataFrame, freq: str | None = None) -> pd.DatetimeIndex:
    """
    Return timestamps where the series has missing bars vs expected regular grid.

    If ``freq`` is None, infer from median bar spacing.
    """
    out = ensure_datetime_index(df)
    if len(out) < 2:
        return pd.DatetimeIndex([])

    if freq is None:
        deltas = out.index.to_series().diff().dropna()
        freq = pd.tseries.frequencies.to_offset(deltas.median())

    expected = pd.date_range(out.index[0], out.index[-1], freq=freq, tz="UTC")
    missing = expected.difference(out.index)
    return missing


def validate_ohlcv(
    df: pd.DataFrame,
    end_str: str | None = None,
    *,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Apply standard post-load cleanup: dedupe and optional end-date trim.

    Matches ``ist.py`` / README idempotency requirements.
    """
    out = ensure_datetime_index(df)
    if drop_duplicates:
        out = drop_duplicate_timestamps(out)
    if end_str is not None:
        out = trim_to_end_date(out, end_str)
    return out
