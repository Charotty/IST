import numpy as np
import pandas as pd
import pytest

from synchronization.config import SynchronizationConfig
from synchronization.multi_timeframe_engine import MultiTimeframeEngine


def _ohlcv(index: pd.DatetimeIndex, base_price: float = 100.0) -> pd.DataFrame:
    n = len(index)
    close = base_price + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=index,
    )


@pytest.fixture
def mtf_frames():
    base_idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    idx_15m = pd.date_range("2024-01-01", periods=800, freq="15min", tz="UTC")
    idx_4h = pd.date_range("2024-01-01", periods=50, freq="4h", tz="UTC")
    return {
        "base": _ohlcv(base_idx, 50000),
        "15m": _ohlcv(idx_15m, 50000),
        "4h": _ohlcv(idx_4h, 50000),
    }


def test_compute_and_merge_adds_mtf_columns(mtf_frames):
    config = SynchronizationConfig(
        auxiliary_timeframes=["15m", "4h"],
        drop_na_after_merge=True,
    )
    engine = MultiTimeframeEngine(
        config=config,
        frames={"15m": mtf_frames["15m"], "4h": mtf_frames["4h"]},
    )
    merged = engine.compute_and_merge(mtf_frames["base"])
    for col in ("rsi_15m", "ema_slope_15m", "rsi_4h", "adx_4h"):
        assert col in merged.columns
    assert len(merged) <= len(mtf_frames["base"])
    assert len(merged) > 0


def test_merge_index_subset_of_base(mtf_frames):
    config = SynchronizationConfig(drop_na_after_merge=False)
    engine = MultiTimeframeEngine(
        config=config,
        frames={"15m": mtf_frames["15m"], "4h": mtf_frames["4h"]},
    )
    merged = engine.compute_and_merge(mtf_frames["base"])
    assert len(merged) == len(mtf_frames["base"])
    assert merged.index.isin(mtf_frames["base"].index).all()


def test_resample_last_no_future_bar_in_same_hour():
    """Hourly value at T must come from 15m bars with timestamp <= end of hour T."""
    idx_15m = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:15:00",
            "2024-01-01 00:30:00",
            "2024-01-01 00:45:00",
            "2024-01-01 01:00:00",
        ],
        utc=True,
    )
    aux = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 99.0]}, index=idx_15m)
    from synchronization.gap_handler import align_to_base

    aligned = align_to_base(aux, "1h", "ffill")
    assert aligned["val"].loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC")] == 4.0
    assert aligned["val"].loc[pd.Timestamp("2024-01-01 01:00:00", tz="UTC")] == 99.0


def test_missing_frames_raises(mtf_frames):
    engine = MultiTimeframeEngine(config=SynchronizationConfig())
    with pytest.raises(ValueError, match="Missing auxiliary"):
        engine.compute_and_merge(mtf_frames["base"])
