import pandas as pd
import pytest

from data_layer.config import DataLayerConfig, MultiTimeframeConfig
from data_layer.loaders.okx_ohlcv_loader import OKXDataLoader


class _MockExchange:
    rateLimit = 0

    def __init__(self, batches: list[list[list]]):
        self._batches = list(batches)
        self._call = 0

    def parse8601(self, s: str) -> int:
        return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int) -> list[list]:
        if self._call >= len(self._batches):
            return []
        batch = self._batches[self._call]
        self._call += 1
        return [row for row in batch if row[0] >= since]


def _bar(ts: str, o: float = 1.0) -> list:
    ms = int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)
    return [ms, o, o, o, o, 100.0]


def test_pagination_advances_since():
    batches = [
        [_bar("2020-01-01 00:00:00"), _bar("2020-01-01 01:00:00")],
        [_bar("2020-01-01 02:00:00")],
        [],
    ]
    loader = OKXDataLoader(rate_limit=False, exchange_factory=lambda: _MockExchange(batches))
    df = loader.fetch_all_ohlcv("BTC/USDT", "1h", "2020-01-01", "2020-01-01 03:00:00", verbose=False)
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2020-01-01 00:00:00", tz="UTC")


def test_pagination_stops_on_duplicate_last_bar():
    duplicate_bar = _bar("2020-01-01 01:00:00")
    batches = [
        [_bar("2020-01-01 00:00:00"), duplicate_bar],
        [duplicate_bar],
    ]
    loader = OKXDataLoader(rate_limit=False, exchange_factory=lambda: _MockExchange(batches))
    df = loader.fetch_all_ohlcv("BTC/USDT", "1h", "2020-01-01", "2020-01-02", verbose=False)
    assert len(df) == 2


def test_trim_by_end_date():
    batches = [
        [
            _bar("2020-01-01 00:00:00"),
            _bar("2020-01-01 01:00:00"),
            _bar("2020-01-01 02:00:00"),
        ],
    ]
    loader = OKXDataLoader(rate_limit=False, exchange_factory=lambda: _MockExchange(batches))
    df = loader.fetch_all_ohlcv(
        "BTC/USDT", "1h", "2020-01-01", "2020-01-01 01:00:00", verbose=False
    )
    assert len(df) == 2
    assert df.index.max() <= pd.Timestamp("2020-01-01 01:00:00", tz="UTC")


def test_fetch_all_timeframes_respects_config():
    calls: list[str] = []

    class _TrackingExchange(_MockExchange):
        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int) -> list[list]:
            calls.append(timeframe)
            return []

    loader = OKXDataLoader(rate_limit=False, exchange_factory=lambda: _TrackingExchange([]))
    config = DataLayerConfig(
        timeframe="1h",
        multi_timeframe=MultiTimeframeConfig(enabled=True, timeframes=["15m", "4h"]),
    )
    result = loader.fetch_all_timeframes(config)
    assert set(result.keys()) == {"1h", "15m", "4h"}
    assert set(calls) == {"1h", "15m", "4h"}


def test_all_timeframes_skips_duplicate_base():
    config = DataLayerConfig(
        timeframe="1h",
        multi_timeframe=MultiTimeframeConfig(enabled=True, timeframes=["1h", "4h"]),
    )
    assert config.all_timeframes() == ["1h", "4h"]
