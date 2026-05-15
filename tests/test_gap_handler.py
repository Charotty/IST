import numpy as np
import pandas as pd

from synchronization.gap_handler import align_to_base, merge_auxiliary
from synchronization.config import SynchronizationConfig


def _hourly_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_align_to_base_ffill_from_4h():
    """4h bars forward-fill to hourly grid without look-ahead in resample."""
    idx_4h = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    aux = pd.DataFrame({"rsi_4h": [10.0, 20.0, 30.0]}, index=idx_4h)
    aligned = align_to_base(aux, "1h", "ffill")
    assert len(aligned) == 9  # 8 hours between first and last 4h bar + 1
    assert aligned["rsi_4h"].iloc[0] == 10.0
    assert aligned["rsi_4h"].iloc[3] == 10.0
    assert aligned["rsi_4h"].iloc[4] == 20.0
    assert aligned["rsi_4h"].iloc[-1] == 30.0


def test_merge_auxiliary_left_join_length():
    base = pd.DataFrame({"close": np.arange(5.0)}, index=_hourly_index(5))
    aux = {
        "15m": pd.DataFrame(
            {"rsi_15m": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
        )
    }
    cfg = SynchronizationConfig(drop_na_after_merge=False)
    merged = merge_auxiliary(base, aux, cfg)
    assert len(merged) == len(base)
