"""MTF indicator columns (aligned with ``ist.py`` / README)."""

from __future__ import annotations

import pandas as pd

from synchronization.gap_handler import ensure_datetime_index
from synchronization.indicators import adx, ema, rsi


def _require_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime_index(df)
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"Missing OHLCV column: {col}")
    return out


def compute_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    """``rsi_15m``, ``ema_slope_15m`` on 15m bars."""
    ohlcv = _require_ohlcv(df)
    out = pd.DataFrame(index=ohlcv.index)
    out["rsi_15m"] = rsi(ohlcv["close"], length=14)
    out["ema_slope_15m"] = ema(ohlcv["close"], length=20).pct_change()
    return out


def compute_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    """``rsi_4h``, ``adx_4h`` on 4h bars."""
    ohlcv = _require_ohlcv(df)
    out = pd.DataFrame(index=ohlcv.index)
    out["rsi_4h"] = rsi(ohlcv["close"], length=14)
    out["adx_4h"] = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=14)
    return out


FEATURE_BUILDERS: dict[str, callable] = {
    "15m": compute_15m_features,
    "4h": compute_4h_features,
}


def build_features(timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
    if timeframe not in FEATURE_BUILDERS:
        raise KeyError(
            f"No MTF feature builder for {timeframe!r}. "
            f"Known: {sorted(FEATURE_BUILDERS)}"
        )
    return FEATURE_BUILDERS[timeframe](df)
