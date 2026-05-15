import pandas as pd

from data_layer.validators.ohlcv_validator import (
    detect_gaps,
    drop_duplicate_timestamps,
    trim_to_end_date,
    validate_ohlcv,
)


def _sample_df() -> pd.DataFrame:
    ts = pd.to_datetime(
        ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 01:00:00", "2020-01-01 03:00:00"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 2.0, 2.5, 4.0],
            "high": [1.0, 2.0, 2.5, 4.0],
            "low": [1.0, 2.0, 2.5, 4.0],
            "close": [1.0, 2.0, 2.5, 4.0],
            "volume": [10.0, 20.0, 25.0, 40.0],
        }
    )


def test_drop_duplicate_timestamps():
    df = drop_duplicate_timestamps(_sample_df())
    assert len(df) == 3
    assert df.loc[pd.Timestamp("2020-01-01 01:00:00", tz="UTC"), "close"] == 2.5


def test_trim_to_end_date():
    df = trim_to_end_date(drop_duplicate_timestamps(_sample_df()), "2020-01-01 01:00:00")
    assert len(df) == 2
    assert df.index.max() == pd.Timestamp("2020-01-01 01:00:00", tz="UTC")


def test_validate_ohlcv_combined():
    df = validate_ohlcv(_sample_df(), end_str="2020-01-01 02:00:00")
    assert len(df) == 2


def test_detect_gaps_hourly():
    df = drop_duplicate_timestamps(_sample_df())
    gaps = detect_gaps(df, freq="1h")
    assert pd.Timestamp("2020-01-01 02:00:00", tz="UTC") in gaps
