import numpy as np
import pandas as pd

from feature_engineering.config import FeatureEngineeringConfig
from feature_engineering.feature_engine import FeatureEngine
from feature_engineering.feature_manager import FeatureManager
from feature_engineering.microstructure.order_book import OrderBookMicrostructure
from feature_engineering.microstructure.simulator import simulate_l2_features


def _ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.3, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 50.0),
        },
        index=idx,
    )


def test_feature_engine_columns_and_no_nan_after_dropna():
    df = FeatureEngine(_ohlcv()).add_indicators().get_processed_data()
    for col in (
        "ema_fast",
        "ema_slow",
        "ema_slope",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "log_ret",
        "volatility",
        "adx",
    ):
        assert col in df.columns
    assert df.isna().sum().sum() == 0
    assert len(df) > 50


def test_feature_manager_preserves_mtf_columns():
    base = _ohlcv()
    base["rsi_15m"] = 50.0
    base["ema_slope_15m"] = 0.01
    base["rsi_4h"] = 55.0
    base["adx_4h"] = 25.0
    cfg = FeatureEngineeringConfig(microstructure={"mode": "off"})
    out = FeatureManager(cfg).transform(base)
    for col in ("rsi_15m", "ema_slope_15m", "rsi_4h", "adx_4h"):
        assert col in out.columns


def test_order_book_obi_balanced():
    ob = OrderBookMicrostructure(depth=2)
    bids = [[100.0, 10.0], [99.0, 10.0]]
    asks = [[101.0, 10.0], [102.0, 10.0]]
    assert ob.calculate_obi(bids, asks) == 0.0


def test_order_book_obi_bid_heavy():
    ob = OrderBookMicrostructure(depth=1)
    assert ob.calculate_obi([[100.0, 30.0]], [[101.0, 10.0]]) == 0.5


def test_simulate_l2_adds_columns():
    df = FeatureEngine(_ohlcv(80)).add_indicators().get_processed_data()
    out = simulate_l2_features(df, seed=42)
    assert "order_book_imbalance" in out.columns
    assert "bid_ask_spread" in out.columns
    assert out["bid_ask_spread"].min() >= 0.0001
